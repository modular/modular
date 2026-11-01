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
// Negative control for the VMM defragmenting allocator: with VMM DISABLED (this
// binary's target env does not set memory_manager_vmm) and memory-manager-only
// mode (no driver backdoor), the same fragmented-pool scenario that VMM defrag
// satisfies in MemoryManagerVMMTest must FAIL here -- there is no single
// contiguous free block large enough and no room to grow a fresh chunk. This
// pins down that the success in the VMM test is genuinely due to defrag.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <memory>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;

namespace {

constexpr size_t kGiB = size_t(1024) * 1024 * 1024;

class MemoryManagerVMMControlTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping VMM control test";
#if MODULAR_MACOS
    GTEST_SKIP() << "skipping on macOS";
#endif
  }

  ErrorOr<DeviceRef> dev() { return Device::create(Device::gpuAPI); }

  std::unique_ptr<MemoryManager> makeCache(Device &device, size_t budgetBytes) {
    ErrorOr<Device::MemInfo> memInfo = device.getMemoryInfo();
    if (memInfo.isError())
      return nullptr;
    return std::unique_ptr<MemoryManager>(MemoryManager::onDevice(
        device, *memInfo, device.allocationAlignment(),
        /*memoryLimit=*/static_cast<int64_t>(budgetBytes)));
  }
};

// Same fragmentation as the VMM test, but with VMM off the 1.5 GiB request
// cannot be satisfied and must return an error.
TEST_F(MemoryManagerVMMControlTest, FragmentedRequestFailsWithoutVmm) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 4 * kGiB;
  const size_t kChunk = 1 * kGiB;
  const size_t kBig = 3 * kGiB / 2;

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();

  std::vector<void *> bufs;
  for (int i = 0; i < 4; ++i) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kChunk, mode));
    bufs.push_back(p);
  }
  EXPECT_NO_ERR(fl.free(bufs[0]));
  EXPECT_NO_ERR(fl.free(bufs[2]));

  ErrorOr<void *> big = fl.allocate(kBig, mode);
  EXPECT_ERR(big);

  EXPECT_NO_ERR(fl.free(bufs[1]));
  EXPECT_NO_ERR(fl.free(bufs[3]));
  cache->clear();
}

} // namespace
