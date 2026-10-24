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
// Condition 4 of the device-context OOM matrix (GEX-3891): side-channel
// exhaustion. The memory manager's own accounting shows ample budget headroom,
// but the physical device memory has been consumed by an allocation the MM does
// not track (a raw `Device::chunkAlloc`, standing in for any out-of-band
// allocator). The MM's next allocation therefore fails at the driver
// (`vmmCreate`) despite the MM believing it has room -- distinguishing a
// device-level shortage from the MM's budget being full (condition 1).
//
// Unlike conditions 1-3 this test cannot be made cheap: reproducing "device out
// of memory while the MM has headroom" genuinely requires consuming almost all
// device memory, because the budget env caps only the MM's budget, not the
// device's physical capacity. It runs with NO budget cap (MM budget ~= 90% of
// free) and reserves a whole GPU (test.resources:gpu-memory in BUILD) so it
// never runs alongside other GPU tests; it skips if the GPU is not near-empty
// at start.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace M::Driver;
using M::Error;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;
constexpr size_t kGiB = size_t(1024) * kMiB;

class MemoryManagerOomSideChannelTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping side-channel OOM test";
#if MODULAR_MACOS
    GTEST_SKIP()
        << "VMM not supported on macOS, skipping side-channel OOM test";
#else
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping side-channel test";
#endif
  }
};

TEST_F(MemoryManagerOomSideChannelTest, UntrackedAllocationExhaustsDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  Device &device = ctx->device();

  // Raw device free/total (driver query, not the MM's budget view).
  ASSIGN_NO_ERR(Device::MemInfo, mem, device.getMemoryInfo());
  if (mem.free < (mem.total * 8) / 10)
    GTEST_SKIP() << "GPU is not near-empty (free=" << mem.free
                 << ", total=" << mem.total
                 << "); another process is resident, skipping to avoid a "
                    "spurious failure";

  // Leave a margin smaller than the MM allocation we will attempt, so the MM
  // has plenty of budget headroom but the device cannot satisfy the request.
  const size_t margin = 256 * kMiB;
  const size_t request = 1 * kGiB;
  ASSERT_GT(mem.free, margin + request);

  auto freeNow = [&device]() -> size_t {
    ErrorOr<Device::MemInfo> m = device.getMemoryInfo();
    return m.isError() ? 0 : m->free;
  };

  // Consume the physical memory out-of-band in chunks the MM does not track.
  // Step the size down as the driver starts refusing: a single coarse size
  // stops as much as one whole chunk short, which would leave more free than
  // `request` and let the allocation under test succeed.
  std::vector<void *> sinks;
  for (size_t chunk = 4 * kGiB; chunk >= 32 * kMiB; chunk /= 2) {
    while (freeNow() > margin) {
      void *p = nullptr;
      if (device.chunkAlloc(&p, chunk, /*isHost=*/false).isError())
        break; // too coarse now; drop to the next size down
      sinks.push_back(p);
    }
  }

  // The condition this test exists to produce is "the device cannot serve the
  // request"; assert it against the driver rather than inferring it from what
  // the loop managed to consume.
  const size_t freeAfterConsume = freeNow();
  if (freeAfterConsume >= request) {
    for (void *p : sinks)
      EXPECT_NO_ERR(device.chunkFree(p, /*isHost=*/false));
    GTEST_SKIP() << "could not drive device free memory below the request "
                    "(free="
                 << freeAfterConsume << ", request=" << request
                 << "); cannot set up the side-channel condition";
  }

  // The MM believes it has ~90% of the original free memory as budget, but the
  // device is now nearly empty, so backing this request fails at the driver.
  ErrorOr<DeviceBufferRef> blocked = ctx->enqueueCreateBuffer(request);
  EXPECT_TRUE(blocked.isError())
      << "expected OOM: device physical exhausted by an untracked allocation "
         "while the MM still had budget headroom";
  EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("(unaccounted)"));

  // The MM's own accounting shows ample headroom (nothing in use), yet the
  // allocation failed -- the shortfall is outside the manager.
  VmmOomSnapshot snap =
      ctx->device().memoryManager().oomSnapshotForTesting(request);
  EXPECT_GE(snap.available(), request);
  EXPECT_EQ(snap.usedBytes, 0u);

  for (void *p : sinks)
    EXPECT_NO_ERR(device.chunkFree(p, /*isHost=*/false));
  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
