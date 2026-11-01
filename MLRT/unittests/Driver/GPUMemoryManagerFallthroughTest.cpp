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

// Regression test for GEX-2965: memory leak from mismatched sync alloc /
// async free in the memory manager's fallthrough allocation path.
//
// The BUILD target configures a small memory manager cache (4MB). Allocations
// larger than 4MB bypass the chunk cache and fall through to the device driver
// (ctx_->memAlloc). The fix ensures this path uses async allocation to match
// the async free in ctx_->memFree(), so both operate on the same stream-ordered
// memory pool and freed memory is properly reused.
//
// Detection strategy: with matching async alloc/free on the same CUDA stream,
// the stream-ordered allocator reuses virtual addresses across alloc/free
// cycles of the same size. With mismatched sync alloc (cuMemAlloc) / async
// free (cuMemFreeAsync), each sync allocation draws from the default heap
// (producing a new address every time) while frees go to the stream-ordered
// pool and are never reclaimed. We detect this by counting unique pointers
// across repeated alloc/free cycles.

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <set>

using namespace M::Driver;
using M::ErrorOr;

namespace {

class GPUMemoryManagerFallthroughTest : public ::testing::Test {
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

  ErrorOr<DeviceContextRef> createTestDeviceContext() {
    return DeviceContext::create(Device::gpuAPI);
  }
};

// Verify that the fallthrough allocation path reuses memory by checking that
// repeated same-size alloc/free cycles return the same pointer. With the fix
// (async alloc matching async free), CUDA's stream-ordered allocator reuses
// addresses. With the bug (sync alloc / async free), each cuMemAlloc returns
// a fresh address from the default heap while cuMemFreeAsync deposits into
// the stream-ordered pool, so addresses are never reused.
TEST_F(GPUMemoryManagerFallthroughTest,
       DISABLED_FallthroughAllocReusesSameAddress) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  MemoryManagerFreeList &freeList = ctx->freeList();

  // 8MB exceeds the 4MB maxCacheSize configured in BUILD, triggering the
  // fallthrough allocation path.
  const size_t allocSize = 8 * 1024 * 1024;
  const Device::AllocMode mode = Device::AllocMode::SYNC;

  // Warmup: one alloc/free cycle to prime the stream-ordered pool.
  ASSIGN_NO_ERR(void *, warmupPtr, freeList.allocate(allocSize, mode));
  EXPECT_NE(warmupPtr, nullptr);
  EXPECT_NO_ERR(freeList.free(warmupPtr));
  EXPECT_NO_ERR(ctx->synchronize());

  // Perform many alloc/free cycles and track unique pointers.
  std::set<void *> uniquePointers;
  const int numCycles = 20;

  for (int i = 0; i < numCycles; ++i) {
    ASSIGN_NO_ERR(void *, ptr, freeList.allocate(allocSize, mode));
    EXPECT_NE(ptr, nullptr);
    uniquePointers.insert(ptr);
    EXPECT_NO_ERR(freeList.free(ptr));
  }
  EXPECT_NO_ERR(ctx->synchronize());

  // With the fix (async alloc + async free on the same stream), the
  // stream-ordered allocator reuses addresses for same-size allocations,
  // producing very few unique pointers (typically 1-2).
  //
  // Without the fix (sync alloc + async free), every cycle produces a new
  // address from the default heap, giving ~50 unique pointers.
  EXPECT_LE(uniquePointers.size(), 5u)
      << "Expected pointer reuse from stream-ordered memory pool, but got "
      << uniquePointers.size() << " unique pointers across " << numCycles
      << " alloc/free cycles. This suggests the fallthrough path is using "
      << "sync allocation (cuMemAlloc) instead of async allocation "
      << "(cuMemAllocAsync), causing a pool mismatch with the async free "
      << "(GEX-2965).";
}

// Same check with multiple DeviceContexts (each with their own CUDA stream).
// Within each context, pointer reuse should occur independently.
TEST_F(GPUMemoryManagerFallthroughTest,
       DISABLED_MultipleContextsFallthroughReusesSameAddress) {
  const size_t allocSize = 8 * 1024 * 1024;
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const int numContexts = 4;
  const int numCyclesPerContext = 8;

  std::vector<DeviceContextRef> contexts;
  for (int i = 0; i < numContexts; ++i) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
    contexts.push_back(std::move(ctx));
  }

  // Warmup all contexts.
  for (auto &ctx : contexts) {
    ASSIGN_NO_ERR(void *, ptr, ctx->freeList().allocate(allocSize, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(ctx->freeList().free(ptr));
    EXPECT_NO_ERR(ctx->synchronize());
  }

  // Track unique pointers per context.
  std::vector<std::set<void *>> perContextPointers(numContexts);

  for (int cycle = 0; cycle < numCyclesPerContext; ++cycle) {
    for (int c = 0; c < numContexts; ++c) {
      ASSIGN_NO_ERR(void *, ptr,
                    contexts[c]->freeList().allocate(allocSize, mode));
      EXPECT_NE(ptr, nullptr);
      perContextPointers[c].insert(ptr);
      EXPECT_NO_ERR(contexts[c]->freeList().free(ptr));
    }
  }

  for (auto &ctx : contexts) {
    EXPECT_NO_ERR(ctx->synchronize());
  }

  // Each context should show pointer reuse within its own stream.
  for (int c = 0; c < numContexts; ++c) {
    EXPECT_LE(perContextPointers[c].size(), 5u)
        << "Context " << c << ": expected pointer reuse but got "
        << perContextPointers[c].size() << " unique pointers across "
        << numCyclesPerContext << " cycles (GEX-2965).";
  }
}

} // namespace
