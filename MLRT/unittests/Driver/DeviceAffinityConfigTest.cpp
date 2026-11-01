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
// Kept in its own binary: runtime.device_task_cpu_ids is read once and cached
// for the process, so the env var must be set before any other test can
// trigger that read. The BUILD target supplies it.
//
// The configured list is "3,9999" against a pool of 8 workers: device 0 names
// a valid worker, device 1 names one past the end of the pool.

#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"

#include "gtest/gtest.h"

using namespace M::MLRT;

namespace {

constexpr size_t kNumWorkers = 8;

// A configured ID that fits the pool is used as-is, ahead of the later tiers.
TEST(DeviceAffinityConfig, UsesConfiguredWorkerIdInRange) {
  EXPECT_EQ(taskIdForDevice(0, kNumWorkers), 3);
}

// A worker ID is an index into the worker array, so a configured ID past the
// end of the pool is an ill-formed mapping. Report it rather than silently
// placing the device somewhere else.
TEST(DeviceAffinityConfigDeathTest, RejectsConfiguredWorkerIdPastPool) {
  EXPECT_DEATH(taskIdForDevice(1, kNumWorkers), "is not a valid worker ID");
}

} // namespace
