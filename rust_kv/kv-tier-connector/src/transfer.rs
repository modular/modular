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
//! The completion handle a lane increments and the main thread polls.
//!
//! The Rust equivalent of the Python `CountingTransfer`: each block's copy
//! marks the [`Counter`] as it lands, and the (Python) main thread polls
//! `is_complete` — a plain atomic read, never touching a CUDA event.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct CounterInner {
    total: usize,
    done: AtomicUsize,
    complete: AtomicBool,
}

/// Completes once every block has passed its device-gating stage.
///
/// The `TieredConnector` equivalent of `CountingTransfer`: each block's H2D (a
/// load) or D2H (an offload) calls [`mark_one`]; complete when the count reaches
/// `total`.
#[derive(Clone)]
pub struct Counter {
    inner: Arc<CounterInner>,
}

impl Counter {
    #[must_use]
    pub fn new(total: usize) -> Self {
        let inner = Arc::new(CounterInner {
            total,
            done: AtomicUsize::new(0),
            complete: AtomicBool::new(total == 0),
        });
        Counter { inner }
    }

    /// Whether all blocks have been marked. Never blocks.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.inner.complete.load(Ordering::Acquire)
    }

    /// Reports one block past the device-gating stage.
    pub fn mark_one(&self) {
        let prev = self.inner.done.fetch_add(1, Ordering::AcqRel);
        if prev + 1 >= self.inner.total {
            self.inner.complete.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_completes_when_all_marked() {
        let c = Counter::new(3);
        assert!(!c.is_complete());
        c.mark_one();
        c.mark_one();
        assert!(!c.is_complete());
        c.mark_one();
        assert!(c.is_complete());
    }

    #[test]
    fn counter_zero_total_is_complete() {
        assert!(Counter::new(0).is_complete());
    }
}
