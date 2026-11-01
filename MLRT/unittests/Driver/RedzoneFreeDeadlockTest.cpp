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
// Redzone validation reads the guard bytes back from the device, which means a
// copy and a stream synchronize. DeviceContext::synchronize() polls pending
// events, and a deferred free-list flush handler takes the memory manager lock,
// so validating while holding that lock self-deadlocks on a non-recursive
// mutex. The invariant under test: freeing a redzoned device allocation must
// not depend on whether a flush handler happens to be pending.
//
// Treat a timeout as a test failure due to deadlock, not a flake.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

using namespace M::Driver;
using M::ErrorOr;

namespace {

class RedzoneFreeDeadlockTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available";
#if MODULAR_MACOS
    GTEST_SKIP() << "GPU memory manager disabled on macOS";
#endif
  }
};

TEST_F(RedzoneFreeDeadlockTest, FreeValidatesRedzonesWithPendingFlushHandler) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  if (!ctx->isDebugAllocatorEnabled("out-of-bounds"))
    GTEST_SKIP() << "redzone checks are off; set "
                    "MODULAR_DEBUG_DEVICE_ALLOCATOR=out-of-bounds";

  // Device memory: host allocations read their redzones directly, never
  // reaching the synchronize this test is about.
  MemoryManagerFreeList &freeList = ctx->freeList();
  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Allocating synchronizes to write the guard pattern, which drains pending
  // events, so both allocations must precede either free.
  ASSIGN_NO_ERR(void *, first, freeList.allocate(4096, mode));
  ASSIGN_NO_ERR(void *, second, freeList.allocate(4096, mode));

  // Enqueues the deferred flush handler.
  EXPECT_NO_ERR(freeList.free(first));

  // Nothing has polled that handler, so this free's redzone read-back runs it
  // from inside synchronize().
  EXPECT_NO_ERR(freeList.free(second));

  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
