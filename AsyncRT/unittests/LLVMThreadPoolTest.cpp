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

#include "AsyncRT/CompilerSupport/LLVMThreadPool.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <thread>

using namespace M;
using namespace M::AsyncRT;

namespace {

CPUDeviceRef makeCPUDevice() {
  CPUDeviceOptions options;
  options.numThreads = 4;
  return getOrCreateCPUDevice(CPUDeviceSource::Test, options);
}

/// The pool-wide turnstile counts every enqueue, so wait() must not return
/// until the enqueued work has run. Tasks go through a task group because the
/// group map is keyed by group pointer and must not see a null key.
TEST(LLVMThreadPoolTest, WaitBlocksUntilEnqueuedTasksFinish) {
  CPUDeviceRef cpuDevice = makeCPUDevice();
  LLVMThreadPool pool(*cpuDevice);
  std::atomic<bool> finished{false};
  {
    llvm::ThreadPoolTaskGroup group(pool);
    group.async([&finished] {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      finished.store(true, std::memory_order_release);
    });
    pool.wait();
    EXPECT_TRUE(finished.load(std::memory_order_acquire));
  }
}

TEST(LLVMThreadPoolTest, WaitCoversEveryTaskGroup) {
  CPUDeviceRef cpuDevice = makeCPUDevice();
  LLVMThreadPool pool(*cpuDevice);
  constexpr unsigned numTasks = 10;
  std::atomic<unsigned> completed{0};
  {
    llvm::ThreadPoolTaskGroup firstGroup(pool);
    llvm::ThreadPoolTaskGroup secondGroup(pool);
    for (unsigned task = 0; task < numTasks; ++task) {
      auto &group = task % 2 == 0 ? firstGroup : secondGroup;
      group.async(
          [&completed] { completed.fetch_add(1, std::memory_order_acq_rel); });
    }
    pool.wait();
    EXPECT_EQ(completed.load(std::memory_order_acquire), numTasks);
  }
}

} // namespace
