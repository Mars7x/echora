//! Staged user-mutex wait queues for the serialized guest-thread runtime.
//!
//! Queue transfer and priority donation follow Atmosphere 6b0c077b4c608a3a2
//! (`kern_k_condition_variable.cpp` and `kern_k_thread.cpp`). The runtime owns
//! address checks, the actual mutex word, thread states and object references.
//! This module never reads memory, schedules a CPU, or validates raw user tags
//! as handles. Self/cyclic waits and duplicate lock records for one owner/VA
//! are explicit unsupported cases, rather than fabricated kernel errors.

use crate::runtime::Trap;
use std::collections::{BTreeMap, BTreeSet};

const MAX_WAIT_RECORDS: usize = 129;
const HAS_WAITERS: u32 = 0x4000_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThreadPriority {
    pub base: u8,
    pub effective: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MutexReturn {
    pub original_owner: u64,
    pub result: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiter {
    owner: u64,
    address: u64,
    tag: u32,
    original_owner: u64,
    order: u64,
    /// A signaled condition-variable waiter's own deadline: it may still
    /// time out while it waits for the mutex (none for ArbitrateLock).
    deadline: Option<std::time::Instant>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MutexWaits {
    waiting: BTreeMap<u64, Waiter>,
    returns: BTreeMap<u64, MutexReturn>,
    next_order: u64,
    /// Waiters with a deadline: while zero the deadline scans return at
    /// once.
    timed: usize,
}

/// What an unlock changed: the effective priorities, the waiter given the
/// mutex and the word to store.
#[derive(Debug)]
pub(crate) struct Unlock {
    pub priorities: BTreeMap<u64, u8>,
    pub selected: Option<u64>,
    pub next_word: u32,
}

fn invalid(message: impl Into<String>) -> Trap {
    Trap::InvalidState {
        message: message.into(),
    }
}

/// Thread priorities as a plan sees them: the runtime's own (`lookup`, one
/// thread at a time) with the plan's changes over them. Only the threads a
/// plan touches are ever looked up.
struct Staged<'a> {
    lookup: &'a dyn Fn(u64) -> Option<ThreadPriority>,
    changes: BTreeMap<u64, ThreadPriority>,
}

impl Staged<'_> {
    fn get(&self, id: u64) -> Result<ThreadPriority, Trap> {
        match self.changes.get(&id) {
            Some(&priority) => Ok(priority),
            None => {
                let priority = (self.lookup)(id).filter(|_| id != 0).ok_or_else(|| {
                    invalid("mutex graph requires a live nonzero thread identity")
                })?;
                if priority.base > 63 || priority.effective > priority.base {
                    return Err(invalid(
                        "mutex priority metadata is outside the modeled 0..63 range",
                    ));
                }
                Ok(priority)
            }
        }
    }

    fn set_effective(&mut self, id: u64, effective: u8) -> Result<(), Trap> {
        let mut priority = self.get(id)?;
        priority.effective = effective;
        self.changes.insert(id, priority);
        Ok(())
    }

    /// The effective priorities that differ from the runtime's: every other
    /// thread keeps its priority, so applying them costs nothing when
    /// inheritance moved nothing.
    fn changed(self) -> BTreeMap<u64, u8> {
        let lookup = self.lookup;
        self.changes
            .into_iter()
            .filter(|&(id, priority)| {
                lookup(id).map(|old| old.effective) != Some(priority.effective)
            })
            .map(|(id, priority)| (id, priority.effective))
            .collect()
    }
}

impl MutexWaits {
    fn take_order(&mut self) -> Result<u64, Trap> {
        let order = self.next_order;
        self.next_order = order
            .checked_add(1)
            .ok_or_else(|| invalid("prototype mutex insertion-order counter exhausted"))?;
        Ok(order)
    }

    // Mirror RestorePriority's owner-chain order. A waiting thread whose
    // effective priority changes is removed/reinserted at its new priority's
    // tail. Unchanged priorities stop propagation and do not consume an order.
    fn restore_priority(&mut self, start: u64, staged: &mut Staged) -> Result<(), Trap> {
        let mut current = Some(start);
        let mut visited = BTreeSet::new();
        while let Some(id) = current {
            if !visited.insert(id) {
                return Err(invalid("prototype mutex priority graph contains a cycle"));
            }
            let mut next_priority = staged.get(id)?.base;
            for (&waiter_id, waiter) in &self.waiting {
                if waiter.owner == id {
                    next_priority = next_priority.min(staged.get(waiter_id)?.effective);
                }
            }
            if next_priority == staged.get(id)?.effective {
                break;
            }
            current = self.waiting.get(&id).map(|waiter| waiter.owner);
            if current.is_some() {
                let order = self.take_order()?;
                self.waiting
                    .get_mut(&id)
                    .expect("id is a current waiter")
                    .order = order;
            }
            staged.set_effective(id, next_priority)?;
        }
        Ok(())
    }

    /// SetBasePriority then RestorePriority from `thread` up its owner chain,
    /// in place. Returns the effective priorities that change.
    pub(crate) fn set_base_priority(
        &mut self,
        thread: u64,
        base: u8,
        lookup: &dyn Fn(u64) -> Option<ThreadPriority>,
    ) -> Result<BTreeMap<u64, u8>, Trap> {
        if base > 63 {
            return Err(invalid("base priority is outside the modeled 0..63 range"));
        }
        let mut staged = Staged {
            lookup,
            changes: BTreeMap::new(),
        };
        let mut priority = staged.get(thread)?;
        priority.base = base;
        staged.changes.insert(thread, priority);
        self.restore_priority(thread, &mut staged)?;
        Ok(staged.changed())
    }

    /// `waiter` waits for `owner`'s mutex at `address`, in place. Returns the
    /// effective priorities that change (`owner`'s chain inherits).
    pub(crate) fn wait(
        &mut self,
        waiter: u64,
        owner: u64,
        address: u64,
        tag: u32,
        lookup: &dyn Fn(u64) -> Option<ThreadPriority>,
    ) -> Result<BTreeMap<u64, u8>, Trap> {
        let mut staged = Staged {
            lookup,
            changes: BTreeMap::new(),
        };
        staged.get(waiter)?;
        staged.get(owner)?;
        if self.waiting.contains_key(&waiter) || self.returns.contains_key(&waiter) {
            return Err(invalid("thread already has a mutex wait or pending return"));
        }
        if self.waiting.len() + self.returns.len() >= MAX_WAIT_RECORDS {
            return Err(invalid("prototype mutex wait/return record limit reached"));
        }
        let mut cursor = Some(owner);
        let mut visited = BTreeSet::new();
        while let Some(id) = cursor {
            if id == waiter || !visited.insert(id) {
                return Err(invalid(
                    "prototype does not implement self or cyclic mutex waits",
                ));
            }
            cursor = self.waiting.get(&id).map(|entry| entry.owner);
        }
        let order = self.take_order()?;
        self.waiting.insert(
            waiter,
            Waiter {
                owner,
                address,
                tag,
                original_owner: owner,
                order,
                deadline: None,
            },
        );
        self.restore_priority(owner, &mut staged)?;
        Ok(staged.changed())
    }

    /// `owner` releases its mutex at `address`, in place: the best waiter
    /// (effective priority, then FIFO) is selected and inherits the rest.
    pub(crate) fn unlock(
        &mut self,
        owner: u64,
        address: u64,
        lookup: &dyn Fn(u64) -> Option<ThreadPriority>,
    ) -> Result<Unlock, Trap> {
        let mut staged = Staged {
            lookup,
            changes: BTreeMap::new(),
        };
        staged.get(owner)?;
        let mut selected: Option<(u8, u64, u64)> = None;
        for (&id, waiter) in &self.waiting {
            if waiter.owner == owner && waiter.address == address {
                let key = (staged.get(id)?.effective, waiter.order, id);
                if selected.is_none_or(|best| key < best) {
                    selected = Some(key);
                }
            }
        }
        let Some((_, _, selected)) = selected else {
            return Ok(Unlock {
                priorities: BTreeMap::new(),
                selected: None,
                next_word: 0,
            });
        };
        let transfers_remaining = self.waiting.iter().any(|(&id, waiter)| {
            id != selected && waiter.owner == owner && waiter.address == address
        });
        if transfers_remaining
            && self
                .waiting
                .values()
                .any(|waiter| waiter.owner == selected && waiter.address == address)
        {
            return Err(invalid(
                "prototype cannot transfer a second lock record for the same owner/address",
            ));
        }
        let selected_waiter = self
            .waiting
            .remove(&selected)
            .expect("the selected thread is waiting");
        if selected_waiter.deadline.is_some() {
            self.timed -= 1;
        }
        self.returns.insert(
            selected,
            MutexReturn {
                original_owner: selected_waiter.original_owner,
                result: None,
            },
        );
        let mut has_waiters = false;
        for waiter in self.waiting.values_mut() {
            if waiter.owner == owner && waiter.address == address {
                waiter.owner = selected;
                // RemoveWaiterByKey transfers the existing tree unchanged.
                // Retain insertion order and the original SVC owner reference.
                has_waiters = true;
            }
        }
        self.restore_priority(owner, &mut staged)?;
        self.restore_priority(selected, &mut staged)?;
        Ok(Unlock {
            priorities: staged.changed(),
            selected: Some(selected),
            next_word: selected_waiter.tag | if has_waiters { HAS_WAITERS } else { 0 },
        })
    }

    /// Give a waiter (a condition-variable waiter moved here when signaled)
    /// the deadline of its condition wait.
    pub(crate) fn set_deadline(
        &mut self,
        id: u64,
        deadline: Option<std::time::Instant>,
    ) -> Result<(), Trap> {
        let waiter = self
            .waiting
            .get_mut(&id)
            .ok_or_else(|| invalid("a mutex deadline requires a mutex waiter"))?;
        let had = waiter.deadline.is_some();
        waiter.deadline = deadline;
        match (had, deadline.is_some()) {
            (false, true) => self.timed += 1,
            (true, false) => self.timed -= 1,
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn earliest_deadline(&self) -> Option<std::time::Instant> {
        if self.timed == 0 {
            return None;
        }
        self.waiting
            .values()
            .filter_map(|waiter| waiter.deadline)
            .min()
    }

    /// Waiters whose deadline has passed at `now`, in id order.
    pub(crate) fn expired(&self, now: std::time::Instant) -> Vec<u64> {
        if self.timed == 0 {
            return Vec::new();
        }
        self.waiting
            .iter()
            .filter(|(_, waiter)| waiter.deadline.is_some_and(|deadline| now >= deadline))
            .map(|(&id, _)| id)
            .collect()
    }

    /// A waiter's deadline passed (Atmosphere: a signaled condition waiter
    /// still waiting for its mutex when its timer fires), in place. It
    /// leaves its owner's waiters, whose inherited priority is restored
    /// without it, and returns TimedOut without the mutex; the mutex word
    /// is left as it is. The owner reference it holds is released when it
    /// resumes, as for a waiter an unlock selected. Returns the effective
    /// priorities that change.
    pub(crate) fn time_out(
        &mut self,
        id: u64,
        lookup: &dyn Fn(u64) -> Option<ThreadPriority>,
    ) -> Result<BTreeMap<u64, u8>, Trap> {
        if self
            .waiting
            .get(&id)
            .is_none_or(|waiter| waiter.deadline.is_none())
        {
            return Err(invalid("only a mutex waiter with a deadline can time out"));
        }
        let mut staged = Staged {
            lookup,
            changes: BTreeMap::new(),
        };
        let waiter = self.waiting.remove(&id).expect("checked above");
        self.timed -= 1;
        self.returns.insert(
            id,
            MutexReturn {
                original_owner: waiter.original_owner,
                result: Some(0xea01),
            },
        );
        self.restore_priority(waiter.owner, &mut staged)?;
        Ok(staged.changed())
    }

    pub(crate) fn has_waiters(&self, owner: u64) -> bool {
        self.waiting.values().any(|waiter| waiter.owner == owner)
    }

    pub(crate) fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub(crate) fn pending_return(&self, id: u64) -> Option<MutexReturn> {
        self.returns.get(&id).copied()
    }

    pub(crate) fn set_wait_result(&mut self, id: u64, result: u32) -> Result<(), Trap> {
        let pending = self
            .returns
            .get_mut(&id)
            .ok_or_else(|| invalid("mutex wake result has no pending return"))?;
        if pending.result.is_some() {
            return Err(invalid("mutex wake result was already recorded"));
        }
        pending.result = Some(result);
        Ok(())
    }

    pub(crate) fn remove_pending_return(&mut self, id: u64) -> Result<(), Trap> {
        if self
            .returns
            .get(&id)
            .is_none_or(|pending| pending.result.is_none())
        {
            return Err(invalid(
                "mutex return cannot retire before its result is recorded",
            ));
        }
        self.returns.remove(&id);
        Ok(())
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        let waiters: Vec<_> = self
            .waiting
            .iter()
            .map(|(&id, waiter)| {
                serde_json::json!({
                    "thread_id":id, "owner_thread_id":waiter.owner,
                    "original_owner_thread_id":waiter.original_owner,
                    "address":waiter.address, "requester_tag":waiter.tag,
                    "insertion_order":waiter.order,
                })
            })
            .collect();
        let pending_returns: Vec<_> = self
            .returns
            .iter()
            .map(|(&id, pending)| {
                serde_json::json!({
                    "thread_id":id, "original_owner_thread_id":pending.original_owner,
                    "result":pending.result,
                })
            })
            .collect();
        serde_json::json!({
            "model":"bounded_acyclic_user_mutex_wait_graph",
            "record_limit":MAX_WAIT_RECORDS,
            "waiting_count":self.waiting.len(), "pending_return_count":self.returns.len(),
            "original_owner_reference_count":self.waiting.len()+self.returns.len(),
            "next_insertion_order":self.next_order,
            "queue_identity":"current_owner_and_process_virtual_address",
            "priority_inheritance":"minimum_effective_priority_along_owner_chains",
            "equal_priority_order":"insertion_fifo_with_reinsert_on_effective_priority_change",
            "remaining_waiter_order_on_handoff":"preserved",
            "self_or_cyclic_waits_implemented":false,
            "duplicate_owner_address_lock_records_implemented":false,
            "wait_cancellation_implemented":false,
            "owns_guest_memory_or_thread_lifetimes":false,
            "horizon_scheduler_implemented":false,
            "waiters":waiters, "pending_returns":pending_returns,
        })
    }
}

/// Completed cancellation of an empty-set, untimed synchronization wait.
/// The runtime writes these values into the saved SVC return context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ZeroHandleReturn {
    pub result: u32,
    pub index: i32,
    pub timeout_ns: i64,
    /// WaitSynchronization returns W1 (the index); a condition-variable wait
    /// returns only W0 and leaves X1 untouched.
    pub writes_index: bool,
}

/// Independent records for WaitSynchronization with no objects and a negative
/// timeout. Per-thread cancellation latches, CPU state and object lifetimes are
/// owned by the runtime; this store only separates parked and returning waits.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ZeroHandleWaits {
    waiting: BTreeMap<u64, WaitRecord>,
    returns: BTreeMap<u64, ZeroHandleReturn>,
    next_condition_order: u64,
    /// Parked waits with a deadline, so the scheduler can skip timer
    /// checks without scanning every wait.
    timed: usize,
}

/// A parked WaitSynchronization: the waited handles in array order and, for
/// a positive timeout, the host-monotonic deadline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WaitRecord {
    pub timeout_ns: i64,
    pub handles: Vec<u32>,
    pub deadline: Option<std::time::Instant>,
    /// `Some` for WaitProcessWideKeyAtomic: the condition variable this
    /// thread waits on and the mutex it reacquires when signaled.
    pub condition: Option<ConditionWait>,
    /// SleepThread: no objects, completes with no result when the deadline
    /// passes, and CancelSynchronization does not affect it.
    pub sleep: bool,
    /// `Some` for WaitForAddress: the arbitrated address and FIFO order.
    pub arbiter: Option<ArbiterWait>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArbiterWait {
    pub address: u64,
    pub order: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConditionWait {
    pub key: u64,
    pub mutex: u64,
    pub tag: u32,
    pub order: u64,
}

impl ZeroHandleWaits {
    pub(crate) fn begin(&mut self, id: u64, timeout_ns: i64) -> Result<(), Trap> {
        if id == 0 || timeout_ns >= 0 {
            return Err(invalid(
                "zero-handle untimed wait requires a nonzero thread and negative timeout",
            ));
        }
        if self.waiting.contains_key(&id) || self.returns.contains_key(&id) {
            return Err(invalid(
                "thread already has a zero-handle wait or pending return",
            ));
        }
        if self.waiting.len() + self.returns.len() >= MAX_WAIT_RECORDS {
            return Err(invalid(
                "prototype zero-handle wait/return record limit reached",
            ));
        }
        self.insert_waiting(
            id,
            WaitRecord {
                timeout_ns,
                handles: Vec::new(),
                deadline: None,
                condition: None,
                sleep: false,
                arbiter: None,
            },
        );
        Ok(())
    }

    /// Park a condition-variable wait (WaitProcessWideKeyAtomic).
    pub(crate) fn begin_condition(
        &mut self,
        id: u64,
        timeout_ns: i64,
        deadline: Option<std::time::Instant>,
        key: u64,
        mutex: u64,
        tag: u32,
    ) -> Result<(), Trap> {
        if id == 0 || timeout_ns == 0 || (timeout_ns > 0) != deadline.is_some() {
            return Err(invalid(
                "condition wait requires a nonzero thread and timeout",
            ));
        }
        if self.waiting.contains_key(&id) || self.returns.contains_key(&id) {
            return Err(invalid("thread already has a wait or pending return"));
        }
        if self.waiting.len() + self.returns.len() >= MAX_WAIT_RECORDS {
            return Err(invalid("prototype wait/return record limit reached"));
        }
        let order = self.next_condition_order;
        self.next_condition_order = order
            .checked_add(1)
            .ok_or_else(|| invalid("condition wait order counter exhausted"))?;
        self.insert_waiting(
            id,
            WaitRecord {
                timeout_ns,
                handles: Vec::new(),
                deadline,
                condition: Some(ConditionWait {
                    key,
                    mutex,
                    tag,
                    order,
                }),
                sleep: false,
                arbiter: None,
            },
        );
        Ok(())
    }

    /// Park an address-arbiter wait (WaitForAddress).
    pub(crate) fn begin_arbiter(
        &mut self,
        id: u64,
        timeout_ns: i64,
        deadline: Option<std::time::Instant>,
        address: u64,
    ) -> Result<(), Trap> {
        if id == 0 || timeout_ns == 0 || (timeout_ns > 0) != deadline.is_some() {
            return Err(invalid(
                "address wait requires a nonzero thread and timeout",
            ));
        }
        if self.waiting.contains_key(&id) || self.returns.contains_key(&id) {
            return Err(invalid("thread already has a wait or pending return"));
        }
        if self.waiting.len() + self.returns.len() >= MAX_WAIT_RECORDS {
            return Err(invalid("prototype wait/return record limit reached"));
        }
        let order = self.next_condition_order;
        self.next_condition_order = order
            .checked_add(1)
            .ok_or_else(|| invalid("wait order counter exhausted"))?;
        self.insert_waiting(
            id,
            WaitRecord {
                timeout_ns,
                handles: Vec::new(),
                deadline,
                condition: None,
                sleep: false,
                arbiter: Some(ArbiterWait { address, order }),
            },
        );
        Ok(())
    }

    /// Address-arbiter waiters on `address`: (thread, wait).
    pub(crate) fn arbiter_waiters(&self, address: u64) -> Vec<(u64, ArbiterWait)> {
        self.waiting
            .iter()
            .filter_map(|(&id, record)| {
                record
                    .arbiter
                    .filter(|wait| wait.address == address)
                    .map(|wait| (id, wait))
            })
            .collect()
    }

    /// Condition-variable waiters on `key`: (thread, wait, insertion order).
    pub(crate) fn condition_waiters(&self, key: u64) -> Vec<(u64, ConditionWait)> {
        self.waiting
            .iter()
            .filter_map(|(&id, record)| {
                record
                    .condition
                    .filter(|wait| wait.key == key)
                    .map(|wait| (id, wait))
            })
            .collect()
    }

    /// Remove a signaled condition waiter that now waits on its mutex instead.
    /// A signaled condition waiter becomes a mutex waiter: it leaves here,
    /// and its deadline (returned) goes with it.
    pub(crate) fn transfer_condition(
        &mut self,
        id: u64,
    ) -> Result<Option<std::time::Instant>, Trap> {
        match self.waiting.get(&id) {
            Some(record) if record.condition.is_some() => {
                let deadline = record.deadline;
                self.remove_waiting(id);
                Ok(deadline)
            }
            _ => Err(invalid(
                "condition transfer requires a parked condition wait",
            )),
        }
    }

    /// A parked wait of either kind (the block-boundary parking check).
    pub(crate) fn is_parked(&self, id: u64) -> bool {
        self.waiting.contains_key(&id)
    }

    /// Park a wait on unsignaled objects (and/or a positive timeout).
    pub(crate) fn begin_objects(
        &mut self,
        id: u64,
        timeout_ns: i64,
        handles: Vec<u32>,
        deadline: Option<std::time::Instant>,
    ) -> Result<(), Trap> {
        if id == 0 || timeout_ns == 0 || (timeout_ns > 0) != deadline.is_some() {
            return Err(invalid(
                "object wait requires a nonzero thread, a nonzero timeout and a deadline exactly for positive timeouts",
            ));
        }
        if self.waiting.contains_key(&id) || self.returns.contains_key(&id) {
            return Err(invalid("thread already has a wait or pending return"));
        }
        if self.waiting.len() + self.returns.len() >= MAX_WAIT_RECORDS {
            return Err(invalid("prototype wait/return record limit reached"));
        }
        self.insert_waiting(
            id,
            WaitRecord {
                timeout_ns,
                handles,
                deadline,
                condition: None,
                sleep: false,
                arbiter: None,
            },
        );
        Ok(())
    }

    /// SleepThread: park `id` until `deadline`.
    pub(crate) fn begin_sleep(
        &mut self,
        id: u64,
        timeout_ns: i64,
        deadline: std::time::Instant,
    ) -> Result<(), Trap> {
        self.begin_objects(id, timeout_ns.max(1), Vec::new(), Some(deadline))?;
        self.waiting
            .get_mut(&id)
            .expect("id is a current waiter")
            .sleep = true;
        Ok(())
    }

    pub(crate) fn waiters(&self) -> Vec<(u64, WaitRecord)> {
        self.waiting
            .iter()
            .map(|(&id, record)| (id, record.clone()))
            .collect()
    }

    fn insert_waiting(&mut self, id: u64, record: WaitRecord) {
        self.timed += usize::from(record.deadline.is_some());
        if let Some(old) = self.waiting.insert(id, record) {
            self.timed -= usize::from(old.deadline.is_some());
        }
    }

    fn remove_waiting(&mut self, id: u64) {
        if let Some(old) = self.waiting.remove(&id) {
            self.timed -= usize::from(old.deadline.is_some());
        }
    }

    /// Whether any parked wait has a deadline (constant time).
    pub(crate) fn has_deadline(&self) -> bool {
        self.timed != 0
    }

    pub(crate) fn earliest_deadline(&self) -> Option<std::time::Instant> {
        self.waiting
            .values()
            .filter_map(|record| record.deadline)
            .min()
    }

    /// Wake a parked wait with its final result (signaled index or timeout).
    pub(crate) fn complete(
        &mut self,
        id: u64,
        result: u32,
        index: i32,
    ) -> Result<ZeroHandleReturn, Trap> {
        let record = self
            .waiting
            .remove(&id)
            .ok_or_else(|| invalid("wait completion requires a parked wait"))?;
        if self.returns.contains_key(&id) {
            return Err(invalid("wait completion cannot replace a pending return"));
        }
        let completed = ZeroHandleReturn {
            result,
            index,
            timeout_ns: record.timeout_ns,
            writes_index: record.condition.is_none() && !record.sleep && record.arbiter.is_none(),
        };
        self.returns.insert(id, completed);
        Ok(completed)
    }

    /// A parked WaitSynchronization (cancellable by CancelSynchronization).
    pub(crate) fn is_waiting(&self, id: u64) -> bool {
        self.waiting.get(&id).is_some_and(|record| {
            record.condition.is_none() && !record.sleep && record.arbiter.is_none()
        })
    }

    pub(crate) fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub(crate) fn cancel(&mut self, id: u64) -> Result<ZeroHandleReturn, Trap> {
        let timeout_ns = self
            .waiting
            .get(&id)
            .map(|record| record.timeout_ns)
            .ok_or_else(|| invalid("zero-handle cancellation requires a parked wait"))?;
        if self.returns.contains_key(&id) {
            return Err(invalid(
                "zero-handle cancellation cannot replace a pending return",
            ));
        }
        let result = ZeroHandleReturn {
            result: 0xec01,
            index: -1,
            timeout_ns,
            writes_index: true,
        };
        self.remove_waiting(id);
        self.returns.insert(id, result);
        Ok(result)
    }

    pub(crate) fn pending_return(&self, id: u64) -> Option<ZeroHandleReturn> {
        self.returns.get(&id).copied()
    }

    pub(crate) fn remove_pending_return(&mut self, id: u64) -> Result<(), Trap> {
        if !self.returns.contains_key(&id) {
            return Err(invalid("zero-handle wait has no pending return to retire"));
        }
        self.returns.remove(&id);
        Ok(())
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        let waiting_threads: Vec<_> = self
            .waiting
            .iter()
            .map(|(&id, record)| {
                serde_json::json!({
                    "thread_id":id, "timeout_ns":record.timeout_ns,
                    "handles":record.handles,
                })
            })
            .collect();
        let pending_returns: Vec<_> = self
            .returns
            .iter()
            .map(|(&id, pending)| {
                serde_json::json!({
                    "thread_id":id, "timeout_ns":pending.timeout_ns,
                    "result":pending.result, "index":pending.index,
                })
            })
            .collect();
        serde_json::json!({
            "model":"zero_handle_untimed_cancellable_waits",
            "record_limit":MAX_WAIT_RECORDS,
            "waiting_count":self.waiting.len(), "pending_return_count":self.returns.len(),
            "retained_object_reference_count":0, "waited_object_count":0,
            "positive_timeout_timers_implemented":true,
            "object_signaling_implemented":true,
            "owns_guest_memory_or_thread_lifetimes":false,
            "waiting_threads":waiting_threads, "pending_returns":pending_returns,
        })
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, any(target_os = "linux", windows)))]
mod tests {
    use super::*;

    /// The in-place changes as plans on a copy, for checking that a failed
    /// change leaves the graph as it was.
    struct Plan {
        state: MutexWaits,
        priorities: BTreeMap<u64, u8>,
        selected: Option<u64>,
        next_word: u32,
    }

    impl MutexWaits {
        fn plan(
            &self,
            change: impl FnOnce(&mut MutexWaits) -> Result<Unlock, Trap>,
        ) -> Result<Plan, Trap> {
            let mut state = self.clone();
            let unlock = change(&mut state)?;
            Ok(Plan {
                state,
                priorities: unlock.priorities,
                selected: unlock.selected,
                next_word: unlock.next_word,
            })
        }

        fn prepare_wait(
            &self,
            waiter: u64,
            owner: u64,
            address: u64,
            tag: u32,
            priorities: &BTreeMap<u64, ThreadPriority>,
        ) -> Result<Plan, Trap> {
            self.plan(|state| {
                state
                    .wait(waiter, owner, address, tag, &|id| {
                        priorities.get(&id).copied()
                    })
                    .map(only_priorities)
            })
        }

        fn prepare_unlock(
            &self,
            owner: u64,
            address: u64,
            priorities: &BTreeMap<u64, ThreadPriority>,
        ) -> Result<Plan, Trap> {
            self.plan(|state| state.unlock(owner, address, &|id| priorities.get(&id).copied()))
        }

        fn prepare_timeout(
            &self,
            id: u64,
            priorities: &BTreeMap<u64, ThreadPriority>,
        ) -> Result<Plan, Trap> {
            self.plan(|state| {
                state
                    .time_out(id, &|id| priorities.get(&id).copied())
                    .map(only_priorities)
            })
        }
    }

    fn only_priorities(priorities: BTreeMap<u64, u8>) -> Unlock {
        Unlock {
            priorities,
            selected: None,
            next_word: 0,
        }
    }

    fn priorities(values: &[(u64, u8)]) -> BTreeMap<u64, ThreadPriority> {
        values
            .iter()
            .map(|&(id, base)| {
                (
                    id,
                    ThreadPriority {
                        base,
                        effective: base,
                    },
                )
            })
            .collect()
    }

    fn apply_priorities(current: &mut BTreeMap<u64, ThreadPriority>, next: &BTreeMap<u64, u8>) {
        for (&id, &effective) in next {
            current.get_mut(&id).unwrap().effective = effective;
        }
    }

    fn wait(
        state: &mut MutexWaits,
        current: &mut BTreeMap<u64, ThreadPriority>,
        waiter: u64,
        owner: u64,
        address: u64,
        tag: u32,
    ) {
        let plan = state
            .prepare_wait(waiter, owner, address, tag, current)
            .unwrap();
        apply_priorities(current, &plan.priorities);
        *state = plan.state;
    }

    fn unlock(
        state: &mut MutexWaits,
        current: &mut BTreeMap<u64, ThreadPriority>,
        owner: u64,
        address: u64,
    ) -> (Option<u64>, u32) {
        let plan = state.prepare_unlock(owner, address, current).unwrap();
        apply_priorities(current, &plan.priorities);
        *state = plan.state;
        (plan.selected, plan.next_word)
    }

    #[test]
    fn a_timed_out_mutex_waiter_leaves_and_its_donation_ends() {
        // 3 (priority 20) and 2 (40) wait on owner 1 (50); 3 is a signaled
        // condition waiter with a deadline.
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 50), (2, 40), (3, 20)]);
        wait(&mut state, &mut current, 2, 1, 0x8000, 0x202);
        wait(&mut state, &mut current, 3, 1, 0x8000, 0x203);
        let deadline = std::time::Instant::now();
        state.set_deadline(3, Some(deadline)).unwrap();
        assert_eq!(current[&1].effective, 20);
        assert_eq!(state.earliest_deadline(), Some(deadline));
        assert_eq!(state.expired(deadline), vec![3]);
        // A waiter without a deadline (ArbitrateLock) never times out.
        assert!(state.prepare_timeout(2, &current).is_err());

        let plan = state.prepare_timeout(3, &current).unwrap();
        apply_priorities(&mut current, &plan.priorities);
        state = plan.state;
        // The owner keeps only 2's donation; 3 returns TimedOut without
        // the mutex, still holding its owner reference until it resumes.
        assert_eq!(current[&1].effective, 40);
        assert!(state.expired(deadline).is_empty());
        assert_eq!(state.earliest_deadline(), None);
        assert_eq!(
            state.pending_return(3),
            Some(MutexReturn {
                original_owner: 1,
                result: Some(0xea01)
            })
        );
        // The unlock then hands the mutex to 2, the one waiter left.
        assert_eq!(
            unlock(&mut state, &mut current, 1, 0x8000),
            (Some(2), 0x202)
        );
        assert_eq!(current[&1].effective, 50);
    }

    #[test]
    fn unlock_prefers_effective_priority_then_fifo_and_preserves_transfer_order() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 50), (2, 40), (3, 40), (4, 20)]);
        wait(&mut state, &mut current, 2, 1, 0x8000, 0x202);
        wait(&mut state, &mut current, 3, 1, 0x8000, 0x203);
        wait(&mut state, &mut current, 4, 1, 0x8000, 0x204);
        assert_eq!(current[&1].effective, 20);
        let orders = (state.waiting[&2].order, state.waiting[&3].order);
        let next_order = state.next_order;

        assert_eq!(
            unlock(&mut state, &mut current, 1, 0x8000),
            (Some(4), 0x4000_0204)
        );
        assert_eq!(current[&1].effective, 50);
        assert!(!state.has_waiters(1));
        assert!(state.has_waiters(4));
        assert_eq!((state.waiting[&2].order, state.waiting[&3].order), orders);
        assert_eq!(state.next_order, next_order);
        assert_eq!(state.waiting[&2].original_owner, 1);
        assert_eq!(state.waiting[&3].original_owner, 1);
        assert_eq!(
            state.pending_return(4),
            Some(MutexReturn {
                original_owner: 1,
                result: None
            })
        );

        assert_eq!(
            unlock(&mut state, &mut current, 4, 0x8000),
            (Some(2), 0x4000_0202)
        );
        assert_eq!(
            unlock(&mut state, &mut current, 2, 0x8000),
            (Some(3), 0x203)
        );
        assert_eq!(state.waiting_count(), 0);
        assert_eq!(state.pending_return(2).unwrap().original_owner, 1);
        assert_eq!(state.pending_return(3).unwrap().original_owner, 1);
    }

    #[test]
    fn chained_donation_and_restoration_reinsert_at_the_new_priority_tail() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 50), (2, 40), (3, 40), (4, 20), (5, 20)]);
        wait(&mut state, &mut current, 2, 1, 0x8000, 2);
        wait(&mut state, &mut current, 3, 1, 0x8000, 3);
        wait(&mut state, &mut current, 4, 2, 0x9000, 4);
        assert_eq!((current[&1].effective, current[&2].effective), (20, 20));
        wait(&mut state, &mut current, 5, 1, 0x8000, 5);
        assert!(state.waiting[&2].order < state.waiting[&5].order);

        assert_eq!(unlock(&mut state, &mut current, 2, 0x9000), (Some(4), 4));
        assert_eq!((current[&1].effective, current[&2].effective), (20, 40));
        assert!(state.waiting[&3].order < state.waiting[&2].order);
        assert_eq!(
            unlock(&mut state, &mut current, 1, 0x8000),
            (Some(5), 0x4000_0005)
        );
        assert_eq!(current[&1].effective, 50);
        assert_eq!(
            unlock(&mut state, &mut current, 5, 0x8000),
            (Some(3), 0x4000_0003)
        );
        assert_eq!(state.waiting[&2].owner, 3);
        assert_eq!(state.waiting[&2].original_owner, 1);
    }

    #[test]
    fn raw_tags_and_virtual_addresses_remain_independent_of_thread_identity() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 50), (2, 30), (3, 31), (4, 25)]);
        wait(&mut state, &mut current, 2, 1, 0x8000, 0);
        wait(&mut state, &mut current, 3, 1, 0x9000, 0xffff_8000);
        wait(&mut state, &mut current, 4, 3, 0x8000, 0x8000_0001);
        let before = state.clone();
        let plan = state.prepare_unlock(1, 0xa000, &current).unwrap();
        assert_eq!(plan.state, before);
        assert_eq!((plan.selected, plan.next_word), (None, 0));
        assert!(plan.priorities.is_empty());

        assert_eq!(unlock(&mut state, &mut current, 1, 0x8000), (Some(2), 0));
        assert_eq!(state.waiting[&4].owner, 3);
        assert_eq!(
            unlock(&mut state, &mut current, 1, 0x9000),
            (Some(3), 0xffff_8000)
        );
        assert_eq!(
            unlock(&mut state, &mut current, 3, 0x8000),
            (Some(4), 0x8000_0001)
        );
    }

    #[test]
    fn wake_results_retain_original_owner_until_the_svc_return_is_retired() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 44), (2, 28)]);
        wait(&mut state, &mut current, 2, 1, 0x0071_04b7_31a8, 0x205);
        assert!(state.pending_return(2).is_none());
        let queued = state.clone();
        assert!(state.set_wait_result(2, 0).is_err());
        assert_eq!(state, queued);
        unlock(&mut state, &mut current, 1, 0x0071_04b7_31a8);
        let selected = state.clone();
        assert!(state.remove_pending_return(2).is_err());
        assert_eq!(state, selected);
        assert!(state.prepare_wait(2, 1, 0x8000, 0x205, &current).is_err());

        state.set_wait_result(2, 0xd401).unwrap();
        assert_eq!(
            state.pending_return(2),
            Some(MutexReturn {
                original_owner: 1,
                result: Some(0xd401)
            })
        );
        let completed = state.clone();
        assert!(state.set_wait_result(2, 0).is_err());
        assert_eq!(state, completed);
        let report = state.report();
        assert_eq!(report["original_owner_reference_count"], 1);
        assert_eq!(report["pending_returns"][0]["result"], 0xd401);
        assert_eq!(report["pending_returns"][0]["original_owner_thread_id"], 1);
        state.remove_pending_return(2).unwrap();
        assert!(state.pending_return(2).is_none());
        assert_eq!(state.report()["original_owner_reference_count"], 0);
        assert!(state.remove_pending_return(2).is_err());
        wait(&mut state, &mut current, 2, 1, 0x8000, 0x205);
    }

    #[test]
    fn cycles_duplicate_waits_and_unrepresentable_lock_transfers_are_atomic_stops() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 44), (2, 28), (3, 30), (4, 40)]);
        let empty = state.clone();
        assert!(state.prepare_wait(1, 1, 0x8000, 1, &current).is_err());
        assert_eq!(state, empty);
        wait(&mut state, &mut current, 2, 1, 0x8000, 2);
        let queued = state.clone();
        assert!(state.prepare_wait(1, 2, 0x9000, 1, &current).is_err());
        assert!(state.prepare_wait(2, 3, 0x9000, 2, &current).is_err());
        assert_eq!(state, queued);
        wait(&mut state, &mut current, 3, 2, 0x8000, 3);
        // The sole selected waiter does not receive a transferred lock-info,
        // so an existing same-address queue is not a duplicate in this case.
        let single = state.prepare_unlock(1, 0x8000, &current).unwrap();
        assert_eq!((single.selected, single.next_word), (Some(2), 2));
        assert_eq!(single.state.waiting[&3].owner, 2);
        wait(&mut state, &mut current, 4, 1, 0x8000, 4);
        let collision = state.clone();
        assert!(state.prepare_unlock(1, 0x8000, &current).is_err());
        assert_eq!(state, collision);
    }

    #[test]
    fn missing_or_invalid_priority_metadata_cannot_change_the_graph() {
        let state = MutexWaits::default();
        let current = priorities(&[(1, 44), (2, 28)]);
        for (waiter, owner) in [(0, 1), (2, 0), (3, 1), (2, 3)] {
            assert!(
                state
                    .prepare_wait(waiter, owner, 0x8000, 0, &current)
                    .is_err()
            );
        }
        for (base, effective) in [(64, 64), (28, 29), (63, 255)] {
            let mut invalid_priorities = current.clone();
            invalid_priorities.insert(2, ThreadPriority { base, effective });
            assert!(
                state
                    .prepare_wait(2, 1, 0x8000, 0, &invalid_priorities)
                    .is_err()
            );
        }
        let mut staged = state.prepare_wait(2, 1, 0x8000, 2, &current).unwrap().state;
        let mut missing_owner = current.clone();
        missing_owner.remove(&1);
        let before = staged.clone();
        assert!(staged.prepare_unlock(1, 0x8000, &missing_owner).is_err());
        assert_eq!(staged, before);
        assert!(staged.set_wait_result(99, 0).is_err());
        assert_eq!(staged, before);
        assert_eq!(state, MutexWaits::default());
    }

    #[test]
    fn waiting_and_pending_returns_share_the_bounded_record_capacity() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&(1..=132).map(|id| (id, 40)).collect::<Vec<_>>());
        for id in 2..=130 {
            wait(&mut state, &mut current, id, 1, 0x8000, id as u32);
        }
        assert_eq!(state.waiting_count(), 129);
        let full = state.clone();
        assert!(state.prepare_wait(131, 1, 0x9000, 131, &current).is_err());
        assert_eq!(state, full);
        assert_eq!(unlock(&mut state, &mut current, 1, 0x8000).0, Some(2));
        assert_eq!(state.waiting_count(), 128);
        let still_full = state.clone();
        assert!(state.prepare_wait(131, 1, 0x9000, 131, &current).is_err());
        assert_eq!(state, still_full);
        state.set_wait_result(2, 0).unwrap();
        state.remove_pending_return(2).unwrap();
        wait(&mut state, &mut current, 131, 1, 0x9000, 131);
        assert_eq!(state.waiting_count(), 129);
    }

    #[test]
    fn insertion_order_exhaustion_rolls_back_enqueue_and_donation_reinsertion() {
        let mut state = MutexWaits::default();
        let mut current = priorities(&[(1, 50), (2, 40), (3, 20)]);
        wait(&mut state, &mut current, 2, 1, 0x8000, 2);
        state.next_order = u64::MAX - 1;
        let before = state.clone();
        let priority_before = current.clone();
        // The new wait can consume the last available order, but its donation
        // would require reinserting thread2. The whole staged plan must fail.
        assert!(state.prepare_wait(3, 2, 0x9000, 3, &current).is_err());
        assert_eq!(state, before);
        assert_eq!(current, priority_before);

        state.next_order = 1;
        wait(&mut state, &mut current, 3, 2, 0x9000, 3);
        state.next_order = u64::MAX;
        let before = state.clone();
        let priority_before = current.clone();
        // Unlock stages a return before priority restoration needs an order.
        // Failure must retain the original waiter and its entire owner chain.
        assert!(state.prepare_unlock(2, 0x9000, &current).is_err());
        assert_eq!(state, before);
        assert_eq!(current, priority_before);
        assert!(state.pending_return(3).is_none());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(all(test, any(target_os = "linux", windows)))]
mod zero_handle_tests {
    use super::*;

    #[test]
    fn negative_waits_preserve_timeouts_through_cancellation_and_return_retirement() {
        let mut state = ZeroHandleWaits::default();
        for (id, timeout_ns) in [(u64::MAX, i64::MIN), (2, -2), (1, -1)] {
            state.begin(id, timeout_ns).unwrap();
            assert!(state.is_waiting(id));
            assert_eq!(state.pending_return(id), None);
        }
        assert_eq!(state.waiting_count(), 3);
        let cancelled = state.cancel(2).unwrap();
        assert_eq!(
            cancelled,
            ZeroHandleReturn {
                result: 0xec01,
                index: -1,
                timeout_ns: -2,
                writes_index: true,
            }
        );
        assert!(!state.is_waiting(2));
        assert_eq!(state.pending_return(2), Some(cancelled));
        assert_eq!(state.waiting_count(), 2);
        assert_eq!(
            state.report(),
            serde_json::json!({
                "model":"zero_handle_untimed_cancellable_waits",
                "record_limit":129, "waiting_count":2, "pending_return_count":1,
                "retained_object_reference_count":0, "waited_object_count":0,
                "positive_timeout_timers_implemented":true,
                "object_signaling_implemented":true,
                "owns_guest_memory_or_thread_lifetimes":false,
                "waiting_threads":[
                    {"thread_id":1, "timeout_ns":-1, "handles":[]},
                    {"thread_id":u64::MAX, "timeout_ns":i64::MIN, "handles":[]},
                ],
                "pending_returns":[
                    {"thread_id":2, "timeout_ns":-2, "result":0xec01, "index":-1},
                ],
            })
        );
        state.remove_pending_return(2).unwrap();
        assert_eq!(state.pending_return(2), None);
        state.begin(2, i64::MIN).unwrap();
        assert_eq!(state.cancel(2).unwrap().timeout_ns, i64::MIN);
        assert_eq!(state.cancel(u64::MAX).unwrap().timeout_ns, i64::MIN);
        assert_eq!(state.cancel(1).unwrap().timeout_ns, -1);
        assert_eq!(state.waiting_count(), 0);
    }

    #[test]
    fn invalid_or_duplicate_operations_preserve_parked_and_pending_records() {
        let mut state = ZeroHandleWaits::default();
        state.begin(1, -1).unwrap();
        let parked = state.clone();
        for (id, timeout_ns) in [(0, -1), (2, 0), (2, 1), (2, i64::MAX), (1, -2)] {
            assert!(matches!(
                state.begin(id, timeout_ns),
                Err(Trap::InvalidState { .. })
            ));
            assert_eq!(state, parked);
        }
        assert!(state.remove_pending_return(1).is_err());
        assert!(state.cancel(0).is_err());
        assert!(state.cancel(2).is_err());
        assert_eq!(state, parked);

        state.cancel(1).unwrap();
        let pending = state.clone();
        assert!(state.begin(1, -1).is_err());
        assert!(state.cancel(1).is_err());
        assert!(state.remove_pending_return(0).is_err());
        assert!(state.remove_pending_return(2).is_err());
        assert_eq!(state, pending);
        state.remove_pending_return(1).unwrap();
        let retired = state.clone();
        assert!(state.remove_pending_return(1).is_err());
        assert_eq!(state, retired);
    }

    #[test]
    fn pending_returns_continue_to_occupy_the_combined_record_limit() {
        let mut state = ZeroHandleWaits::default();
        for id in 1..=129 {
            state.begin(id, -1).unwrap();
        }
        let full = state.clone();
        assert!(state.begin(130, -1).is_err());
        assert_eq!(state, full);
        state.cancel(1).unwrap();
        assert_eq!(state.waiting_count(), 128);
        assert_eq!(state.pending_return(1).unwrap().result, 0xec01);
        let returning = state.clone();
        assert!(state.begin(130, -1).is_err());
        assert!(state.begin(1, -1).is_err());
        assert!(state.cancel(130).is_err());
        assert_eq!(state, returning);

        state.remove_pending_return(1).unwrap();
        state.begin(130, -2).unwrap();
        assert_eq!(state.waiting_count(), 129);
        assert_eq!(state.report()["pending_return_count"], 0);
        assert!(state.is_waiting(130));
        assert!(!state.is_waiting(1));
    }
}
