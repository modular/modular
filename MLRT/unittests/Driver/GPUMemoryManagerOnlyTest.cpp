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

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <memory>

using namespace M::Driver;
using M::ErrorOr;

namespace {

class GPUMemoryManagerOnlyTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError()) {
      GTEST_SKIP() << "GPU not available, skipping tests";
    }
#if MODULAR_MACOS
    GTEST_SKIP() << "MemoryManager not used on macOS, skipping tests";
#endif
  }

  ErrorOr<DeviceRef> createTestDevice() {
    return Device::create(Device::gpuAPI);
  }

  ErrorOr<DeviceContextRef> createTestDeviceContext() {
    return DeviceContext::create(Device::gpuAPI);
  }
};

TEST_F(GPUMemoryManagerOnlyTest, RejectsAllocationLargerThanMaxCacheSize) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  MemoryManagerFreeList &freeList = ctx->freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  constexpr size_t kMaxCacheSize = 16 * 1024 * 1024;

  EXPECT_ERR_MSG(freeList.allocate(kMaxCacheSize + 1024, mode),
                 "Memory manager cannot satisfy allocation");
}

// TODO: GEX-3475
// Technically, this environment for this test is a contradiction.
// On one hand, we are disabling MemoryManager (via `memoryLimit=0`).
// On the other hand, we are forcing exclusive use of the memory manager
// (via env var in the build rules) and no backdoor allocations from the
// device driver. Ideally, the construction of a MemoryManager instance
// should fail but we allow it and expect failures to happen during allocation.
// That is too late in the application lifecycle.
// Hence, fix the lifecycle by detecting such contradictions early and failing
// fast. And then remove such tests.
TEST_F(GPUMemoryManagerOnlyTest, RejectsWhenGpuMemoryManagerIsDisabled) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());

  std::unique_ptr<MemoryManager> cache(
      MemoryManager::onDevice(*device, memInfo, device->allocationAlignment(),
                              /*memoryLimit=*/0));

  EXPECT_ERR_MSG(cache->freeList()->allocate(64, Device::AllocMode::SYNC),
                 "memory manager is disabled");
}

} // namespace
