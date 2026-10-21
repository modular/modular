//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//! Background transfer lanes on real OS threads.
//!
//! A port of `connectors/transfer_worker.py`'s `TransferLane`, but on a
//! `std::thread` with the Python GIL released instead of a Python thread that
//! contends for it. Each lane is one thread that issues its work (`run`) FIFO,
//! polls all in-flight work every loop, and runs `on_complete` on its own
//! thread once a work item's [`Pollable`] is ready. Slow work never blocks
//! issuing later work, and multi-tier chains (disk-read -> H2D) hop lanes by
//! having `on_complete` submit to the next lane.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender, channel},
    },
    thread::JoinHandle,
    time::Duration,
};

/// How long a lane waits for new work before re-polling in-flight work.
const POLL_INTERVAL: Duration = Duration::from_micros(500);

/// A host-pollable completion handle for one unit of lane work. Created and
/// polled on the lane thread, so it need not be `Send`.
pub trait Pollable {
    /// Whether the work has completed. Never blocks.
    fn is_ready(&self) -> bool;
}

/// A [`Pollable`] that is always ready — for work that completes synchronously
/// inside `run` (e.g. a no-op disk write).
pub struct AlreadyComplete;

impl Pollable for AlreadyComplete {
    fn is_ready(&self) -> bool {
        true
    }
}

type RunFn = Box<dyn FnOnce() -> Box<dyn Pollable> + Send>;
type CompleteFn = Box<dyn FnOnce() + Send>;

struct Work {
    run: RunFn,
    on_complete: CompleteFn,
}

/// A single background thread that issues and polls one kind of work.
///
/// `tx`/`handle` are behind mutexes so `shutdown` works through `&self` — the
/// connector holds lanes in `Arc` (lane `on_complete`s chain to sibling lanes),
/// so shutdown must not require unique ownership.
pub struct TransferLane {
    tx: Mutex<Option<Sender<Option<Work>>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
    // Count of submitted-but-not-yet-completed work items (queued + in-flight),
    // so `wait_until_idle` can block until the lane has fully drained without
    // stopping it (unlike `shutdown`).
    pending: Arc<AtomicUsize>,
}

impl TransferLane {
    /// Spawns the lane thread.
    #[must_use]
    pub fn new(name: &str) -> Self {
        let (tx, rx) = channel::<Option<Work>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pending_worker = Arc::clone(&pending);
        let handle = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || lane_loop(&rx, &pending_worker))
            .expect("spawn transfer lane thread");
        TransferLane {
            tx: Mutex::new(Some(tx)),
            handle: Mutex::new(Some(handle)),
            pending,
        }
    }

    /// Enqueues a unit of work without blocking. `run` issues the work and
    /// returns its [`Pollable`]; `on_complete` runs on the lane thread once the
    /// pollable is ready. A no-op after shutdown.
    pub fn submit(&self, run: RunFn, on_complete: CompleteFn) {
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            self.pending.fetch_add(1, Ordering::AcqRel);
            if tx.send(Some(Work { run, on_complete })).is_err() {
                self.pending.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    /// Blocks until every submitted work item has completed, without stopping
    /// the lane (unlike [`shutdown`]). Used to quiesce in-flight work (e.g. an
    /// offload's D2H + write-through) before a prefix-cache reset.
    pub fn wait_until_idle(&self) {
        while self.pending.load(Ordering::Acquire) > 0 {
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Drains in-flight work and stops the lane thread. Idempotent, `&self`.
    pub fn shutdown(&self) {
        if let Some(tx) = self.tx.lock().unwrap().take() {
            drop(tx.send(None));
        }
        if let Some(handle) = self.handle.lock().unwrap().take() {
            drop(handle.join());
        }
    }
}

impl Drop for TransferLane {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn lane_loop(rx: &Receiver<Option<Work>>, pending: &AtomicUsize) {
    // (pollable, on_complete) for work issued but not yet observed complete.
    let mut inflight: Vec<(Box<dyn Pollable>, CompleteFn)> = Vec::new();
    let mut shutting_down = false;

    loop {
        // `Some(Some(w))` = new work, `Some(None)` = shutdown, `None` = nothing
        // yet (timeout). Block for new work when idle; poll-interval when we
        // have in-flight work to re-check.
        let next: Option<Option<Work>> = if inflight.is_empty() {
            match rx.recv() {
                Ok(Some(w)) => Some(Some(w)),
                // Explicit sentinel or a dropped sender both mean shut down.
                Ok(None) | Err(_) => Some(None),
            }
        } else {
            match rx.recv_timeout(POLL_INTERVAL) {
                Ok(Some(w)) => Some(Some(w)),
                Ok(None) | Err(RecvTimeoutError::Disconnected) => Some(None),
                Err(RecvTimeoutError::Timeout) => None,
            }
        };

        match next {
            Some(Some(work)) => {
                let pollable = (work.run)();
                inflight.push((pollable, work.on_complete));
            }
            Some(None) => shutting_down = true,
            None => {}
        }

        // Poll in-flight work in order; fire on_complete for those ready.
        let mut still: Vec<(Box<dyn Pollable>, CompleteFn)> = Vec::with_capacity(inflight.len());
        for (pollable, on_complete) in inflight.drain(..) {
            if pollable.is_ready() {
                on_complete();
                pending.fetch_sub(1, Ordering::AcqRel);
            } else {
                still.push((pollable, on_complete));
            }
        }
        inflight = still;

        if shutting_down && inflight.is_empty() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use super::*;

    struct Flag(Arc<AtomicBool>);
    impl Pollable for Flag {
        fn is_ready(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn runs_and_completes_in_order() {
        let lane = TransferLane::new("test-lane");
        let counter = Arc::new(AtomicUsize::new(0));

        // Two ready-immediately items; on_complete increments.
        for _ in 0..2 {
            let c = Arc::clone(&counter);
            lane.submit(
                Box::new(|| Box::new(AlreadyComplete)),
                Box::new(move || {
                    c.fetch_add(1, Ordering::AcqRel);
                }),
            );
        }
        // Drain.
        lane.shutdown();
        assert_eq!(counter.load(Ordering::Acquire), 2);
    }

    #[test]
    fn polls_until_ready() {
        let lane = TransferLane::new("test-lane-2");
        let flag = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let f = Arc::clone(&flag);
        let d = Arc::clone(&done);
        lane.submit(
            Box::new(move || Box::new(Flag(f))),
            Box::new(move || d.store(true, Ordering::Release)),
        );
        // Not ready yet.
        std::thread::sleep(Duration::from_millis(5));
        assert!(!done.load(Ordering::Acquire));
        // Flip the flag; the lane should observe it and fire on_complete.
        flag.store(true, Ordering::Release);
        lane.shutdown();
        assert!(done.load(Ordering::Acquire));
    }
}
