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
// The virtual-address failure condition for the single-arena VMM allocator:
// physical memory is available, but no contiguous free address range is large
// enough to carve the request from. The recovery lever is a larger VA
// reservation (memory_manager_vmm_va_factor), not more memory, so this is the
// one condition where the message must not talk about memory pressure.
//
// CUDA-only: the setup needs freed blocks to reach the root free list on a
// synchronize, which does not hold on HIP (GEX-4194).
//
// It needs its own target because it needs `memory_manager_vmm_va_factor` set
// to 1. The arena is normally reserved several times larger than the physical
// budget, so there is always a large unused run at the frontier and the VA
// gate cannot be reached; with a factor of 1 the arena is exactly the budget
// and can be fully carved up.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <utility>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;

// Must match the BUILD env for this target.
constexpr size_t kBudget = 256 * kMiB;
constexpr size_t kPage = 2 * kMiB;

MemoryManager &mm(DeviceContext &ctx) { return ctx.device().memoryManager(); }

class MemoryManagerOomVaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping OOM tests";
#if MODULAR_MACOS
    GTEST_SKIP() << "VMM not supported on macOS, skipping OOM tests";
#else
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping OOM tests";
    // The setup below frees alternate blocks and synchronizes to get them onto
    // the root free list, which is what leaves the free space fragmented but
    // allocatable. On HIP the gating events stay unfired through the
    // synchronize, so the blocks remain child-owned and the shortfall is a
    // pending one rather than an address-space one -- a different condition,
    // correctly classified as `avoidable`. GEX-4194 covers giving AMD its own
    // coverage here.
    if ((*device)->api() != Device::cudaAPI)
      GTEST_SKIP() << "freed blocks do not reach the root free list on a "
                      "synchronize outside CUDA, so this condition cannot be "
                      "constructed here (GEX-4194)";
#endif
  }
};

// Carve the whole arena into equal blocks, then free every other one. Half the
// budget is now free and reclaimable, but every free run is one block wide
// because the blocks either side of it are still live. A request of two blocks
// therefore has ample physical behind it and nowhere to sit.
TEST_F(MemoryManagerOomVaTest, NoContiguousRangeDespiteAvailablePhysical) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  const size_t block = 8 * kMiB;
  const size_t count = kBudget / block; // fills the arena exactly
  const size_t request = 2 * block;     // wider than any free run will be

  // Whole pages per block, so freeing one releases its pages outright. A block
  // sharing a page with a live neighbour would leave those bytes trapped, which
  // is a different condition from the address-space one under test.
  ASSERT_EQ(block % kPage, 0u);

  std::vector<DeviceBufferRef> live;
  for (size_t i = 0; i < count; ++i) {
    ASSIGN_NO_ERR(DeviceBufferRef, b, ctx->enqueueCreateBuffer(block));
    live.push_back(std::move(b));
  }

  // Free alternate blocks and synchronize, so the freed descs migrate to the
  // root free list. Without the sync they stay child-owned and the shortfall
  // would read as pending rather than as an address-space problem.
  for (size_t i = 1; i < count; i += 2)
    live[i].reset();
  EXPECT_NO_ERR(ctx->synchronize());

  // Physical is fine and no lever is needed; there is simply no run wide
  // enough.
  VmmOomSnapshot snap = mm(*ctx).oomSnapshotForTesting(request);
  EXPECT_GE(snap.available(), request)
      << "expected ample allocatable physical from the freed blocks";
  EXPECT_EQ(snap.recoverableBySync(), 0u);
  EXPECT_EQ(snap.trapped(), 0u);
  EXPECT_EQ(snap.largestRootFreeVA, block)
      << "every free run should be exactly one block wide";
  EXPECT_LT(snap.largestRootFreeVA, request);

  ErrorOr<DeviceBufferRef> blocked = ctx->enqueueCreateBuffer(request);
  EXPECT_TRUE(blocked.isError())
      << "expected OOM: physical is available but no contiguous VA range fits";
  if (blocked.isError()) {
    EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("(VA fragmentation)"));
    // The lever is the address-space reservation, not memory.
    EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("va_factor"));
    EXPECT_THAT(blocked.getError(),
                ::testing::Not(::testing::HasSubstr("(out of memory)")));
  } else {
    blocked.takeValue().reset();
  }

  // A single-block request still succeeds, proving the memory is there and the
  // failure was the shape of the free space, not its amount.
  ErrorOr<DeviceBufferRef> fits = ctx->enqueueCreateBuffer(block);
  EXPECT_FALSE(fits.isError())
      << "expected a one-block request to fit in a freed run";
  if (!fits.isError())
    fits.takeValue().reset();

  live.clear();
  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
