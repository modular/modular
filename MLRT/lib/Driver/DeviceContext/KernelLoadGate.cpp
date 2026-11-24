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

#include "MLRT/Driver/DeviceContext/KernelLoadGate.h"

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "llvm/Support/Threading.h"

#include <cassert>

namespace M::Driver {

using AsyncRT::AsyncValueRef;
using AsyncRT::Chain;

bool KernelLoadGate::tryAcquireCollective(AsyncRT::CPUDevice &cpuDevice,
                                          AsyncValueRef<Chain> &grant) {
  // Hold lock while updating KernelLoadGate internal state
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ == State::Idle) {
    // Idle -> Active: allocate idle chain for next loader, return true.
    idle_ = AsyncValueRef<Chain>::allocate(cpuDevice);
    if (inFlightLoads_ == 0) {
      state_ = State::Active;
      return true;
    }
    // Idle -> Draining: allocate drain grant for caller and return false.
    state_ = State::Draining;
    drainGrant_ = AsyncValueRef<Chain>::allocate(cpuDevice);
    grant = drainGrant_.copy();
    return false;
  }
  // {Active, Draining}: queue a chain behind the owner and return false.
  queuedCollectives_.push_back(AsyncValueRef<Chain>::allocate(cpuDevice));
  grant = queuedCollectives_.back().copy();
  return false;
}

void KernelLoadGate::releaseCollective() {
  AsyncValueRef<Chain> wake;
  {
    // Hold lock while updating KernelLoadGate internal state
    std::lock_guard<std::mutex> lock(mu_);
    assert(state_ == State::Active &&
           "releasing a kernel-load gate no collective owns");
    if (queuedCollectives_.empty()) {
      // Active -> Idle: mark idle chain ready to wake next loader.
      state_ = State::Idle;
      wake = std::move(idle_);
    } else {
      // Pop the next queued collective's chain.
      AsyncValueRef<Chain> next = std::move(queuedCollectives_.front());
      queuedCollectives_.erase(queuedCollectives_.begin());
      // The owner's own loads close before its enqueues finish, and every other
      // load has waited since the claim, so none can be in flight here.
      assert(inFlightLoads_ == 0 &&
             "handing off a kernel-load gate with loads still open");
      wake = std::move(next);
    }
  }
  // Wake up the next loader outside the critical section.
  if (wake)
    std::move(wake).emplace();
}

void KernelLoadGate::openLoad() {
  while (true) {
    AsyncValueRef<Chain> waitOn;
    {
      // Hold lock while updating KernelLoadGate internal state
      std::lock_guard<std::mutex> lock(mu_);
      // Idle: increment the in-flight load counter and return.
      // Special case: if this thread is enqueuing a collective, it already
      // holds the gate and doesn't need to wait for itself. But we still
      // track the in-flight load.
      if (state_ == State::Idle || isEnqueueThread()) {
        ++inFlightLoads_;
        return;
      }
      // {Active, Draining}: copy idle chain and wait on it.
      waitOn = idle_.copy();
    }
    // Wait for the collective to complete, then try again.
    AsyncRT::await(waitOn);
  }
}

void KernelLoadGate::closeLoad() {
  AsyncValueRef<Chain> grant;
  {
    // Hold lock while updating KernelLoadGate internal state
    std::lock_guard<std::mutex> lock(mu_);
    // Decrement the in-flight load counter.
    --inFlightLoads_;
    if (inFlightLoads_ == 0 && state_ == State::Draining) {
      // Draining -> Active: open the gate for the next collective.
      state_ = State::Active;
      grant = std::move(drainGrant_);
    }
  }
  if (grant)
    std::move(grant).emplace();
}

bool KernelLoadGate::isEnqueueThread() const {
  uint64_t self = llvm::get_threadid();
  for (uint64_t entry : enqueueThreads_)
    if (entry == self)
      return true;
  return false;
}

void KernelLoadGate::beginEnqueue() {
  uint64_t self = llvm::get_threadid();
  std::lock_guard<std::mutex> lock(mu_);
  // A second registration would mean a launch body waited and this thread ran
  // another device's share inside it, which keying on thread identity rules
  // out.
  assert(!isEnqueueThread() &&
         "thread already registered as a collective enqueuer");
  enqueueThreads_.push_back(self);
}

void KernelLoadGate::endEnqueue() {
  uint64_t self = llvm::get_threadid();
  std::lock_guard<std::mutex> lock(mu_);
  for (uint64_t &entry : enqueueThreads_) {
    if (entry != self)
      continue;
    // Order does not matter, so swap with the back rather than shift.
    entry = enqueueThreads_.back();
    enqueueThreads_.pop_back();
    return;
  }
  assert(false && "unbalanced kernel-load gate enqueue scope");
}

} // namespace M::Driver
