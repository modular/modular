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

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Support/Chain.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <thread>

using M::AsyncRT::AsyncValueRef;
using M::AsyncRT::Chain;
using M::Driver::KernelLoadGate;

namespace {

// Long enough that a load which should be waiting has visibly not returned. A
// load that wrongly proceeds returns in microseconds, so this cannot pass by
// accident; it can only make a correct gate look slow.
constexpr std::chrono::milliseconds kSettle{50};

// Opens a load on its own thread. A thread outside the runtime blocks in
// AsyncRT::await rather than donating, which keeps the test deterministic.
class LoadOnThread {
public:
  explicit LoadOnThread(KernelLoadGate &gate)
      : gate_(gate), thread_([this] {
          gate_.openLoad();
          entered_ = true;
        }) {}
  ~LoadOnThread() {
    thread_.join();
    gate_.closeLoad();
  }
  LoadOnThread(const LoadOnThread &) = delete;
  LoadOnThread &operator=(const LoadOnThread &) = delete;

  bool enteredAfterSettling() const {
    std::this_thread::sleep_for(kSettle);
    return entered_;
  }

private:
  KernelLoadGate &gate_;
  std::atomic<bool> entered_{false};
  std::thread thread_;
};

class KernelLoadGateTest : public ::testing::Test {
protected:
  void SetUp() override {
    runtime_ = M::AsyncRT::getOrCreateCPUDevice(
        M::AsyncRT::CPUDeviceSource::Test,
        M::AsyncRT::CPUDeviceOptions().withMainWillNotDonate());
  }
  M::AsyncRT::CPUDevice &cpu() { return *runtime_; }

  M::AsyncRT::CPUDeviceRef runtime_;
  KernelLoadGate gate_;
};

TEST_F(KernelLoadGateTest, IdleGateGrantsImmediately) {
  AsyncValueRef<Chain> grant;
  EXPECT_TRUE(gate_.tryAcquireCollective(cpu(), grant));
  EXPECT_FALSE(grant);
  gate_.releaseCollective();

  EXPECT_TRUE(gate_.tryAcquireCollective(cpu(), grant));
  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, QueuedCollectiveIsHandedTheGate) {
  AsyncValueRef<Chain> first, second;
  ASSERT_TRUE(gate_.tryAcquireCollective(cpu(), first));
  ASSERT_FALSE(gate_.tryAcquireCollective(cpu(), second));
  EXPECT_FALSE(second.isReady());

  gate_.releaseCollective();
  EXPECT_TRUE(second.isReady());

  gate_.releaseCollective();
  AsyncValueRef<Chain> third;
  EXPECT_TRUE(gate_.tryAcquireCollective(cpu(), third));
  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, QueuedCollectivesAreGrantedInArrivalOrder) {
  AsyncValueRef<Chain> owner, second, third;
  ASSERT_TRUE(gate_.tryAcquireCollective(cpu(), owner));
  ASSERT_FALSE(gate_.tryAcquireCollective(cpu(), second));
  ASSERT_FALSE(gate_.tryAcquireCollective(cpu(), third));

  gate_.releaseCollective();
  EXPECT_TRUE(second.isReady());
  EXPECT_FALSE(third.isReady());

  gate_.releaseCollective();
  EXPECT_TRUE(third.isReady());
  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, CollectiveWaitsForInFlightLoadsToDrain) {
  gate_.openLoad();
  gate_.openLoad();

  AsyncValueRef<Chain> grant;
  ASSERT_FALSE(gate_.tryAcquireCollective(cpu(), grant));
  EXPECT_FALSE(grant.isReady());

  gate_.closeLoad();
  EXPECT_FALSE(grant.isReady());
  gate_.closeLoad();
  EXPECT_TRUE(grant.isReady());

  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, LoadWaitsForCollectiveToRelease) {
  AsyncValueRef<Chain> grant;
  ASSERT_TRUE(gate_.tryAcquireCollective(cpu(), grant));

  LoadOnThread load(gate_);
  EXPECT_FALSE(load.enteredAfterSettling());

  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, LoadWaitsForDrainingCollective) {
  gate_.openLoad();
  AsyncValueRef<Chain> grant;
  ASSERT_FALSE(gate_.tryAcquireCollective(cpu(), grant));

  // A load arriving while the collective drains waits behind it, or a steady
  // trickle of loads could hold the collective off indefinitely.
  LoadOnThread late(gate_);
  EXPECT_FALSE(late.enteredAfterSettling());

  gate_.closeLoad();
  ASSERT_TRUE(grant.isReady());
  EXPECT_FALSE(late.enteredAfterSettling());

  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, LoadsWaitThroughHandOff) {
  AsyncValueRef<Chain> first, second;
  ASSERT_TRUE(gate_.tryAcquireCollective(cpu(), first));
  ASSERT_FALSE(gate_.tryAcquireCollective(cpu(), second));

  LoadOnThread load(gate_);
  EXPECT_FALSE(load.enteredAfterSettling());

  gate_.releaseCollective();
  ASSERT_TRUE(second.isReady());
  EXPECT_FALSE(load.enteredAfterSettling());

  gate_.releaseCollective();
}

TEST_F(KernelLoadGateTest, EnqueueThreadLoadsPassTheirOwnCollective) {
  AsyncValueRef<Chain> grant;
  ASSERT_TRUE(gate_.tryAcquireCollective(cpu(), grant));

  gate_.beginEnqueue();
  gate_.openLoad();
  gate_.closeLoad();
  gate_.endEnqueue();

  gate_.releaseCollective();
}

} // namespace
