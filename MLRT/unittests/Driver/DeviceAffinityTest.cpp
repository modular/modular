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

#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

#include <set>

using namespace M::MLRT;

namespace {

constexpr size_t kNumWorkers = 384;

// MXSERV-189: taskIdForNonDevice built a lookup table bounded by INT_MAX
// instead of by the worker count, so its first call burned ~7.2s of CPU and
// 8.59GB of resident memory. It is called from a CUDA host callback, which
// blocks the stream, so a slow first call parks the GPU.
//
TEST(DeviceAffinity, NonDeviceIdsAreInRangeAndSkipWorkerZero) {
  for (int taskId = 0; taskId < static_cast<int>(kNumWorkers) * 3; ++taskId) {
    int id = taskIdForNonDevice(taskId, kNumWorkers);
    if (id == kDefaultTaskId)
      continue;
    EXPECT_GE(id, 0);
    EXPECT_LT(id, static_cast<int>(kNumWorkers));
    // Worker 0 is the MEF executor thread and is never a valid target.
    EXPECT_NE(id, 0);
  }
}

// The point of the non-device mapping is to spread work rather than funnel it
// onto one thread; a table bounded by the wrong value could collapse the range.
TEST(DeviceAffinity, DistinctTaskIdsSpreadAcrossWorkers) {
  std::set<int> assigned;
  for (int taskId = 0; taskId < static_cast<int>(kNumWorkers); ++taskId) {
    int id = taskIdForNonDevice(taskId, kNumWorkers);
    if (id != kDefaultTaskId)
      assigned.insert(id);
  }
  // Exact count depends on how many IDs are reserved for device threads on the
  // host, which varies with GPU count and config, so assert a loose floor.
  EXPECT_GT(assigned.size(), kNumWorkers / 2);
}

// Tier 3 assigns device workers contiguously upward from worker 1, and
// taskIdForNonDevice allocates downward from the top of the pool, so the two
// only meet once every other worker is spoken for. Check the regime that
// matters: ordinary load, well short of exhausting the pool.
//
// Only real device IDs matter. Production callers pass an actual device index,
// so IDs beyond the device count are never requested.
TEST(DeviceAffinity, NonDeviceStaysAwayFromDeviceWorkers) {
  int numDevices =
      M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);
  if (numDevices == 0)
    GTEST_SKIP() << "no GPUs present";

  // A pool smaller than the physical core count also exercises the round-robin
  // tier, which is where the two allocators could meet.
  for (size_t numWorkers : {size_t(16), kNumWorkers}) {
    std::set<int> deviceWorkers;
    for (int deviceId = 0; deviceId < numDevices; ++deviceId) {
      int id = taskIdForDevice(deviceId, numWorkers);
      if (id != kDefaultTaskId)
        deviceWorkers.insert(id);
    }

    for (int taskId = 0; taskId < static_cast<int>(numWorkers) / 2; ++taskId) {
      int id = taskIdForNonDevice(taskId, numWorkers);
      if (id == kDefaultTaskId)
        continue;
      EXPECT_EQ(deviceWorkers.count(id), 0u)
          << "numWorkers=" << numWorkers << " taskId=" << taskId
          << " landed on device worker " << id;
    }
  }
}

// A worker ID is an index into the worker array, so anything >= numWorkers is
// an out-of-bounds access. The config-supplied and NUMA-inferred tiers can both
// name IDs above the pool size.
TEST(DeviceAffinity, DeviceTaskIdStaysWithinWorkerRange) {
  for (size_t numWorkers : {size_t(2), size_t(8), size_t(16), kNumWorkers}) {
    for (int deviceId = 0; deviceId < 8; ++deviceId) {
      int id = taskIdForDevice(deviceId, numWorkers);
      if (id == kDefaultTaskId)
        continue;
      EXPECT_GE(id, 0);
      EXPECT_LT(id, static_cast<int>(numWorkers))
          << "numWorkers=" << numWorkers << " deviceId=" << deviceId;
    }
  }
}

} // namespace
