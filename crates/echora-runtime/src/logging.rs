//! Bounded, in-memory capture for the initial LM/ILogger implementation.
//!
//! The wire header and malformed-size success/drop behavior follow Atmosphere
//! revision 6b0c077b4c608a3a2ee712b8dc35fee87ceb62e8, specifically
//! `libraries/libstratosphere/source/lm/impl/lm_log_packet_header.hpp` and
//! `libraries/libstratosphere/source/lm/srv/lm_logger_impl.cpp`. The latter also
//! accepts every destination mask and stores it globally. This local capture is
//! independent of that requested routing mask; it does not deliver to UART or
//! Target Manager and performs no file or network I/O.
//!
//! Header integers are decoded as little endian, matching the modeled guest.
//! Raw flags, padding, severity, and verbosity are preserved without imposing
//! additional wire-format restrictions. In particular, an unset endian flag
//! does not switch this decoder to big endian. Payloads remain opaque because
//! a chunk may continue in another packet. Unlike the server's ring-buffer
//! drops, exhaustion of this prototype's configured retention limits is an
//! explicit error. A caller must check its guest reply buffer before invoking
//! `log` if later reply failure must leave capture state unchanged.

use anyhow::{Context, Result, bail};
use serde::{Serialize, Serializer};
use serde_json::{Value, json};

pub const LOG_PACKET_HEADER_SIZE: usize = 24;
pub const DESTINATION_TARGET_MANAGER: u32 = 1;
pub const DESTINATION_UART: u32 = 2;
pub const DESTINATION_UART_IF_SLEEP: u32 = 4;
pub const DESTINATION_ALL: u32 = 0xffff;

/// Retention limits for the local diagnostic sink, not console ABI limits.
/// Byte accounting includes each complete 24-byte header and its payload.
/// Record count additionally bounds per-record allocation/metadata overhead.
/// Zero limits are permitted and can prevent retention of any valid packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LogLimits {
    pub max_packet_bytes: usize,
    pub max_records: usize,
    pub max_retained_bytes: usize,
}

impl Default for LogLimits {
    fn default() -> Self {
        Self {
            max_packet_bytes: 64 * 1024,
            max_records: 1024,
            max_retained_bytes: 1024 * 1024,
        }
    }
}

/// Identity supplied by the trusted logger session, never by the log packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoggerContext {
    process_id: u64,
}

impl LoggerContext {
    pub fn new(process_id: u64) -> Self {
        Self { process_id }
    }

    pub fn process_id(&self) -> u64 {
        self.process_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogDisposition {
    Stored,
    DroppedMalformed,
}

/// Validated header fields and an opaque payload. `process_id` is the effective
/// sender identity; `untrusted_packet_process_id` exists only for diagnostics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LogRecord {
    pub process_id: u64,
    pub untrusted_packet_process_id: u64,
    pub thread_id: u64,
    pub flags: u8,
    pub reserved: u8,
    pub severity: u8,
    pub verbosity: u8,
    pub payload_size: u32,
    pub destination_mask: u32,
    #[serde(rename = "payload_hex", serialize_with = "serialize_payload_hex")]
    pub payload: Vec<u8>,
}

fn serialize_payload_hex<S: Serializer>(payload: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::Error;

    const HEX: &[u8; 16] = b"0123456789abcdef";
    let capacity = payload
        .len()
        .checked_mul(2)
        .ok_or_else(|| S::Error::custom("log payload hex length overflow"))?;
    let mut encoded = String::new();
    encoded
        .try_reserve_exact(capacity)
        .map_err(S::Error::custom)?;
    for &byte in payload {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 15)] as char);
    }
    serializer.serialize_str(&encoded)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MalformedReason {
    HeaderTooShort,
    PayloadLengthMismatch {
        declared_payload_bytes: u32,
        expected_packet_bytes: usize,
    },
    PayloadLengthOverflow {
        declared_payload_bytes: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MalformedPacket {
    process_id: u64,
    actual_packet_bytes: usize,
    reason: MalformedReason,
}

pub struct LocalLogSink {
    limits: LogLimits,
    destination: u32,
    records: Vec<LogRecord>,
    retained_packet_bytes: usize,
    malformed_drop_count: u64,
    last_malformed: Option<MalformedPacket>,
}

impl Default for LocalLogSink {
    fn default() -> Self {
        Self::with_limits(LogLimits::default())
    }
}

impl LocalLogSink {
    pub fn with_limits(limits: LogLimits) -> Self {
        Self {
            limits,
            destination: DESTINATION_TARGET_MANAGER,
            records: Vec::new(),
            retained_packet_bytes: 0,
            malformed_drop_count: 0,
            last_malformed: None,
        }
    }

    /// Exposed so the IPC layer can reject a guest buffer before allocating it.
    pub fn max_packet_bytes(&self) -> usize {
        self.limits.max_packet_bytes
    }

    /// The requested routing mask is global to this sink, not per logger.
    /// Every u32 is accepted, including zero and unknown destination bits.
    pub fn set_destination(&mut self, destination: u32) {
        self.destination = destination;
    }

    pub fn destination(&self) -> u32 {
        self.destination
    }

    pub fn records(&self) -> &[LogRecord] {
        &self.records
    }

    /// Store one structurally valid packet or count an explicit malformed drop.
    /// Size/record/retained-byte limits return an error without changing logical
    /// capture state. Malformed packets do not consume the retention budget.
    pub fn log(&mut self, logger: &LoggerContext, packet: &[u8]) -> Result<LogDisposition> {
        if packet.len() > self.limits.max_packet_bytes {
            bail!(
                "local log packet is {} bytes, exceeding the {}-byte limit",
                packet.len(),
                self.limits.max_packet_bytes
            );
        }
        if packet.len() < LOG_PACKET_HEADER_SIZE {
            return self.drop_malformed(logger, packet.len(), MalformedReason::HeaderTooShort);
        }

        // The fixed header has been checked before any of these reads.
        let payload_size = u32::from_le_bytes(packet[20..24].try_into().unwrap());
        let expected_length = usize::try_from(payload_size)
            .ok()
            .and_then(|size| LOG_PACKET_HEADER_SIZE.checked_add(size));
        let Some(expected_length) = expected_length else {
            return self.drop_malformed(
                logger,
                packet.len(),
                MalformedReason::PayloadLengthOverflow {
                    declared_payload_bytes: payload_size,
                },
            );
        };
        if expected_length != packet.len() {
            return self.drop_malformed(
                logger,
                packet.len(),
                MalformedReason::PayloadLengthMismatch {
                    declared_payload_bytes: payload_size,
                    expected_packet_bytes: expected_length,
                },
            );
        }

        if self.records.len() >= self.limits.max_records {
            bail!(
                "local log record limit of {} is exhausted",
                self.limits.max_records
            );
        }
        let retained_packet_bytes = self
            .retained_packet_bytes
            .checked_add(packet.len())
            .context("local log retained-byte count overflow")?;
        if retained_packet_bytes > self.limits.max_retained_bytes {
            bail!(
                "local log retention would use {retained_packet_bytes} bytes, exceeding the {}-byte limit",
                self.limits.max_retained_bytes
            );
        }

        // All configured limits are checked before allocating or committing a
        // record. Allocator failure is also explicit, with no record/counter
        // changes. Reserved Vec capacity is not externally observable state.
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(packet.len() - LOG_PACKET_HEADER_SIZE)
            .context("cannot allocate local log payload")?;
        payload.extend_from_slice(&packet[LOG_PACKET_HEADER_SIZE..]);
        self.records
            .try_reserve(1)
            .context("cannot allocate local log record")?;
        let record = LogRecord {
            process_id: logger.process_id,
            untrusted_packet_process_id: u64::from_le_bytes(packet[0..8].try_into().unwrap()),
            thread_id: u64::from_le_bytes(packet[8..16].try_into().unwrap()),
            flags: packet[16],
            reserved: packet[17],
            severity: packet[18],
            verbosity: packet[19],
            payload_size,
            destination_mask: self.destination,
            payload,
        };
        self.records.push(record);
        self.retained_packet_bytes = retained_packet_bytes;
        Ok(LogDisposition::Stored)
    }

    fn drop_malformed(
        &mut self,
        logger: &LoggerContext,
        actual_packet_bytes: usize,
        reason: MalformedReason,
    ) -> Result<LogDisposition> {
        let next_count = self
            .malformed_drop_count
            .checked_add(1)
            .context("local log malformed-drop counter is exhausted")?;
        self.last_malformed = Some(MalformedPacket {
            process_id: logger.process_id,
            actual_packet_bytes,
            reason,
        });
        self.malformed_drop_count = next_count;
        Ok(LogDisposition::DroppedMalformed)
    }

    pub fn report(&self) -> Value {
        json!({
            "backend": "bounded_local_diagnostic_capture",
            "header_decoding": "little_endian",
            "raw_endian_flag_controls_decoding": false,
            "payload_decoding": "opaque_bytes_may_contain_continuation_chunks",
            "sender_identity": "logger_session_process_id",
            "destination_policy": "global_requested_mask_recorded_independent_local_capture",
            "external_destination_delivery": false,
            "malformed_policy": "service_success_with_counted_drop",
            "resource_limit_policy": "explicit_error_without_record_or_counter_mutation",
            "destination_mask": self.destination,
            "limits": self.limits,
            "record_count": self.records.len(),
            "retained_packet_bytes": self.retained_packet_bytes,
            "malformed_drop_count": self.malformed_drop_count,
            "last_malformed": self.last_malformed,
            "records": self.records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(wire_pid: u64, thread_id: u64, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0; LOG_PACKET_HEADER_SIZE];
        bytes[0..8].copy_from_slice(&wire_pid.to_le_bytes());
        bytes[8..16].copy_from_slice(&thread_id.to_le_bytes());
        bytes[16] = 7; // Head, Tail, LittleEndian.
        bytes[18] = 2;
        bytes[19] = 3;
        bytes[20..24].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn header_offsets_and_trusted_session_identity_are_preserved() {
        let logger = LoggerContext::new(0x1234_5678_90ab_cdef);
        let wire_pid = 0xfedc_ba09_8765_4321;
        let thread_id = 0x1122_3344_5566_7788;
        let mut bytes = packet(wire_pid, thread_id, &[0x80, 0xff, 0x00]);
        bytes[16] = 0xa3;
        bytes[17] = 0x5a;
        bytes[18] = 0xf4;
        bytes[19] = 0x8d;
        let original = bytes.clone();
        let mut sink = LocalLogSink::default();

        assert_eq!(sink.log(&logger, &bytes).unwrap(), LogDisposition::Stored);
        assert_eq!(bytes, original, "the borrowed guest snapshot is unchanged");
        assert_eq!(
            sink.records(),
            &[LogRecord {
                process_id: logger.process_id(),
                untrusted_packet_process_id: wire_pid,
                thread_id,
                flags: 0xa3,
                reserved: 0x5a,
                severity: 0xf4,
                verbosity: 0x8d,
                payload_size: 3,
                destination_mask: 1,
                payload: vec![0x80, 0xff, 0],
            }]
        );
        assert_eq!(sink.report()["records"][0]["payload_hex"], "80ff00");
        assert_eq!(sink.report()["retained_packet_bytes"], 27);
    }

    #[test]
    fn empty_and_continuation_payloads_need_no_chunk_or_metadata_validation() {
        let logger = LoggerContext::new(7);
        let mut sink = LocalLogSink::default();
        assert_eq!(
            sink.log(&logger, &packet(0, 0, &[])).unwrap(),
            LogDisposition::Stored
        );
        let mut continuation = packet(0, 1, &[0x80]);
        continuation[16] = 0x80; // Unknown bit; no head/tail/endian bit.
        continuation[17] = 0xff;
        continuation[18] = 0xff;
        continuation[19] = 0xff;
        assert_eq!(
            sink.log(&logger, &continuation).unwrap(),
            LogDisposition::Stored
        );
        assert_eq!(sink.records()[1].payload, [0x80]);
        assert_eq!(sink.records()[1].thread_id, 1);
        assert_eq!(sink.records()[1].flags, 0x80);
        assert_eq!(sink.records()[1].reserved, 0xff);
        assert_eq!(sink.records()[1].severity, 0xff);
        assert_eq!(sink.report()["header_decoding"], "little_endian");
        assert_eq!(sink.report()["raw_endian_flag_controls_decoding"], false);
    }

    #[test]
    fn every_short_header_is_a_counted_drop_without_retention() {
        let logger = LoggerContext::new(9);
        let mut sink = LocalLogSink::default();
        for size in 0..LOG_PACKET_HEADER_SIZE {
            assert_eq!(
                sink.log(&logger, &vec![0xa5; size]).unwrap(),
                LogDisposition::DroppedMalformed
            );
        }
        assert!(sink.records().is_empty());
        assert_eq!(sink.retained_packet_bytes, 0);
        assert_eq!(sink.malformed_drop_count, 24);
        assert_eq!(
            sink.last_malformed,
            Some(MalformedPacket {
                process_id: 9,
                actual_packet_bytes: 23,
                reason: MalformedReason::HeaderTooShort,
            })
        );
    }

    #[test]
    fn declared_length_must_match_exactly_including_trailing_bytes() {
        let logger = LoggerContext::new(9);
        let mut sink = LocalLogSink::default();
        for declared in [0_u32, 2, u32::MAX] {
            let mut bytes = packet(123, 456, &[0xaa]);
            bytes[20..24].copy_from_slice(&declared.to_le_bytes());
            assert_eq!(
                sink.log(&logger, &bytes).unwrap(),
                LogDisposition::DroppedMalformed
            );
        }
        let mut truncated_payload = packet(123, 456, &[0xaa]);
        truncated_payload.pop();
        assert_eq!(
            sink.log(&logger, &truncated_payload).unwrap(),
            LogDisposition::DroppedMalformed
        );
        assert_eq!(sink.malformed_drop_count, 4);
        assert!(sink.records().is_empty());
        assert_eq!(sink.retained_packet_bytes, 0);
    }

    #[test]
    fn destination_is_global_accepts_unknown_bits_and_records_receipt_value() {
        let first = LoggerContext::new(1);
        let second = LoggerContext::new(2);
        let mut sink = LocalLogSink::default();
        assert_eq!(sink.destination(), DESTINATION_TARGET_MANAGER);
        for (logger, destination) in [
            (&first, 0),
            (&second, DESTINATION_ALL),
            (&first, 0xdead_beef),
        ] {
            sink.set_destination(destination);
            sink.log(logger, &packet(999, 0, &[])).unwrap();
        }
        assert_eq!(sink.destination(), 0xdead_beef);
        assert_eq!(sink.records()[0].destination_mask, 0);
        assert_eq!(sink.records()[1].destination_mask, 0xffff);
        assert_eq!(sink.records()[2].destination_mask, 0xdead_beef);
        assert_eq!(sink.records()[0].process_id, 1);
        assert_eq!(sink.records()[1].process_id, 2);
        assert_eq!(sink.report()["external_destination_delivery"], false);
    }

    #[test]
    fn oversized_packet_fails_before_even_malformed_drop_mutation() {
        let mut sink = LocalLogSink::with_limits(LogLimits {
            max_packet_bytes: 24,
            ..LogLimits::default()
        });
        let logger = LoggerContext::new(1);
        sink.log(&logger, &[0]).unwrap();
        let before = sink.report();
        assert_eq!(sink.max_packet_bytes(), 24);
        assert!(sink.log(&logger, &packet(0, 0, &[1])).is_err());
        assert_eq!(sink.report(), before);
        assert!(sink.log(&logger, &[0xff; 25]).is_err());
        assert_eq!(sink.report(), before);
    }

    #[test]
    fn record_exhaustion_is_an_error_but_malformed_drops_still_count() {
        let mut sink = LocalLogSink::with_limits(LogLimits {
            max_records: 1,
            ..LogLimits::default()
        });
        let logger = LoggerContext::new(1);
        let bytes = packet(0, 0, &[]);
        sink.log(&logger, &bytes).unwrap();
        let before = sink.report();
        assert!(sink.log(&logger, &bytes).is_err());
        assert_eq!(sink.report(), before);
        assert_eq!(
            sink.log(&logger, &[0]).unwrap(),
            LogDisposition::DroppedMalformed
        );
        assert_eq!(sink.records().len(), 1);
        assert_eq!(sink.retained_packet_bytes, 24);
        assert_eq!(sink.malformed_drop_count, 1);
    }

    #[test]
    fn aggregate_budget_counts_headers_and_allows_the_exact_boundary() {
        let mut sink = LocalLogSink::with_limits(LogLimits {
            max_packet_bytes: 32,
            max_records: 4,
            max_retained_bytes: 49,
        });
        let logger = LoggerContext::new(1);
        sink.log(&logger, &packet(0, 0, &[1])).unwrap(); // 25 bytes.
        sink.log(&logger, &packet(0, 0, &[])).unwrap(); // 24 more bytes.
        assert_eq!(sink.retained_packet_bytes, 49);
        let before = sink.report();
        assert!(sink.log(&logger, &packet(0, 0, &[])).is_err());
        assert_eq!(sink.report(), before);
    }

    #[test]
    fn zero_retention_limits_do_not_fake_success() {
        let mut sink = LocalLogSink::with_limits(LogLimits {
            max_packet_bytes: 24,
            max_records: 0,
            max_retained_bytes: 0,
        });
        let before = sink.report();
        assert!(
            sink.log(&LoggerContext::new(1), &packet(0, 0, &[]))
                .is_err()
        );
        assert_eq!(sink.report(), before);
    }

    #[test]
    fn exhausted_drop_counter_is_an_explicit_error_without_state_change() {
        let mut sink = LocalLogSink {
            malformed_drop_count: u64::MAX,
            ..LocalLogSink::default()
        };
        let before = sink.report();
        assert!(sink.log(&LoggerContext::new(1), &[0]).is_err());
        assert_eq!(sink.report(), before);
    }
}
