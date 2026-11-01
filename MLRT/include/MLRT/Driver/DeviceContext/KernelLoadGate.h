//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Keeps first-time kernel loads from interleaving with a multi-device
// collective's per-device enqueues. The CUDA driver owns the only instance;
// HIP loads do not wait for a busy device, so HIP has none.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_KERNELLOADGATE_H
#define MLRT_DRIVER_DEVICECONTEXT_KERNELLOADGATE_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

namespace M::Driver {

/// Keeps first-time kernel loads from interleaving with a multi-device
/// collective's per-device enqueues.
///
/// A first-time load blocks until its device is idle, holding that context's
/// driver lock. A collective kernel spins at a device-side barrier until every
/// peer arrives, so a load that lands partway through a collective's enqueues
/// can wait on a device that never goes idle while it holds up a peer's
/// enqueue. See MXSERV-441.
///
/// Loads are readers and run concurrently. A collective is exclusive: it waits
/// for in-flight loads to drain, and new loads wait until it, and any
/// collectives queued behind it, have finished enqueueing.
///
/// No wait blocks a worker. Loads wait on an `AsyncValueRef<Chain>` through
/// `AsyncRT::await`, which lets the worker run other tasks, including a
/// collective's per-device enqueue pinned to it. The collective side never
/// waits: `tryAcquireCollective` grants the gate or returns a grant to dispatch
/// from, and whichever thread finishes the last per-device enqueue calls
/// `releaseCollective`. A frame that held the gate across a donating wait
/// could be buried beneath a task that needs the gate.
///
/// All state lives in this object, never at file scope: the two sides of the
/// gate are linked into different shared objects of a served process.
class KernelLoadGate {
public:
  KernelLoadGate() = default;
  KernelLoadGate(const KernelLoadGate &) = delete;
  KernelLoadGate &operator=(const KernelLoadGate &) = delete;

  /// Claims the gate for one collective's enqueues. Never blocks.
  ///
  /// Returns true if the collective owns the gate now. Otherwise returns false
  /// and sets `grant` to a chain that becomes available once it does: after the
  /// collective ahead of it releases, or after in-flight loads drain. Either
  /// way the caller dispatches its enqueues only once it owns the gate, and
  /// must call `releaseCollective` exactly once after the last of them.
  bool tryAcquireCollective(AsyncRT::CPUDevice &cpuDevice,
                            AsyncRT::AsyncValueRef<AsyncRT::Chain> &grant);

  /// Ends the owning collective's claim. Hands the gate to the next queued
  /// collective, or reopens it to loads. Called from whichever thread finishes
  /// the last per-device enqueue, never from a frame waiting on the fan-out.
  void releaseCollective();

  /// Enters a region that must not overlap a collective's enqueues: the
  /// driver calls of a first-time kernel load, or a vendor library call that
  /// can block on a busy device. Waits while any collective holds or is
  /// queued for the gate; concurrent with other such regions.
  ///
  /// Does not wait on a thread registered through `beginEnqueue`, because that
  /// work belongs to the collective holding the gate.
  ///
  /// The paired form for a caller that cannot hold a C++ object across the
  /// region, which is the FFI. C++ callers should prefer `LoadScope`.
  void openLoad();
  void closeLoad();

  /// RAII form of `openLoad` / `closeLoad`.
  class LoadScope {
  public:
    explicit LoadScope(KernelLoadGate &gate) : gate_(gate) { gate_.openLoad(); }
    ~LoadScope() { gate_.closeLoad(); }
    LoadScope(const LoadScope &) = delete;
    LoadScope &operator=(const LoadScope &) = delete;

  private:
    KernelLoadGate &gate_;
  };

  /// Registers the calling thread as running one device's share of the enqueue
  /// work for the collective holding the gate, so its loads proceed instead of
  /// waiting on a gate their own collective owns.
  ///
  /// Keyed on thread identity, since loads reach `loadFunction` from deep
  /// inside kernel code with no parameter to carry the fact. Sound because a
  /// per-device launch body never suspends, so no other task runs on the
  /// thread while it is registered.
  ///
  /// Paired calls rather than a scope object because the only caller is the
  /// FFI.
  void beginEnqueue();
  void endEnqueue();

private:
  enum class State {
    /// No collective holds or is queued for the gate; loads proceed.
    Idle,
    /// A collective has claimed the gate and waits for in-flight loads to
    /// finish before it may enqueue.
    Draining,
    /// A collective owns the gate and is enqueueing.
    Active,
  };

  /// Whether the calling thread is registered through `beginEnqueue`.
  /// Callers hold `mu_`.
  bool isEnqueueThread() const;

  std::mutex mu_;
  State state_ = State::Idle;
  size_t inFlightLoads_ = 0;
  /// Threads running the open collective's per-device enqueues, by
  /// `llvm::get_threadid()`. One entry per device, so a scan is enough.
  llvm::SmallVector<uint64_t, 8> enqueueThreads_;
  /// Fulfilled when the gate returns to Idle; waited on by loads. Survives a
  /// hand-off between queued collectives, so loads wait out the whole run of
  /// them. Null while Idle.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> idle_;
  /// The grant owed to the collective in Draining, fulfilled by the load whose
  /// close brings `inFlightLoads_` to zero. Null outside Draining.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> drainGrant_;
  /// Grants for collectives queued behind the owner, in arrival order. Empty
  /// unless collectives overlap, and short when they do.
  llvm::SmallVector<AsyncRT::AsyncValueRef<AsyncRT::Chain>, 2>
      queuedCollectives_;
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_KERNELLOADGATE_H
