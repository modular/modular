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
// Tests for the VMM (CUDA Virtual Memory Management) defragmenting allocator.
// VMM is enabled process-wide via the target's `env` (the config is parsed once
// per process, so it cannot be toggled per-test). The load-bearing test
// fragments the pool so the largest contiguous free block is smaller than a
// later request; with memory-manager-only mode there is no driver backdoor, so
// a successful allocation can ONLY come from VMM defrag (reserve fresh VA +
// remap scattered free granules). The VMM-off counterpart lives in
// MemoryManagerVMMControlTest.cpp (separate binary). These tests are single-GPU
// (owner-device access only); multi-GPU peer access is a follow-on phase.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <vector>

using namespace M::Driver;
using M::Error;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

constexpr size_t kGiB = size_t(1024) * 1024 * 1024;
constexpr size_t kMiB = size_t(1024) * 1024;

class MemoryManagerVMMTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping VMM tests";
#if MODULAR_MACOS
    GTEST_SKIP() << "VMM not supported on macOS, skipping VMM tests";
#else
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping VMM tests";
#endif
  }

  ErrorOr<DeviceRef> dev() { return Device::create(Device::gpuAPI); }

  // Fresh, independent device MemoryManager with an explicit byte budget so the
  // test neither disturbs nor depends on the device's shared manager. VMM,
  // chunk-percent, memory-manager-only and self-check come from the target env.
  std::unique_ptr<MemoryManager> makeCache(Device &device, size_t budgetBytes) {
    ErrorOr<Device::MemInfo> memInfo = device.getMemoryInfo();
    if (memInfo.isError())
      return nullptr;
    return std::unique_ptr<MemoryManager>(MemoryManager::onDevice(
        device, *memInfo, device.allocationAlignment(),
        /*memoryLimit=*/static_cast<int64_t>(budgetBytes)));
  }
};

// Write a sentinel into the first bytes of every `pageSize`-sized slice of
// [ptr, ptr+size) and read it back, proving the whole range is physically
// backed and accessible (an unbacked granule would fault here).
ErrorOrSuccess writeAndVerify(DeviceContext &ctx, void *ptr, size_t size,
                              size_t pageSize) {
  const std::vector<uint8_t> pattern(64, 0xAB);
  for (size_t off = 0; off < size; off += pageSize) {
    ErrorOrSuccess e = ctx.enqueueCopyToDevice(static_cast<char *>(ptr) + off,
                                               pattern.data(), pattern.size());
    if (e.isError())
      return e.takeError();
  }
  if (ErrorOrSuccess e = ctx.synchronize(); e.isError())
    return e.takeError();

  for (size_t off = 0; off < size; off += pageSize) {
    std::vector<uint8_t> readback(64, 0);
    ErrorOrSuccess e = ctx.enqueueCopyFromDevice(
        readback.data(), static_cast<char *>(ptr) + off, readback.size());
    if (e.isError())
      return e.takeError();
    if (ErrorOrSuccess s = ctx.synchronize(); s.isError())
      return s.takeError();
    if (readback != pattern)
      return Error("VMM buffer read-back mismatch: range not correctly backed");
  }
  return M::success();
}

// Phase 0: an ordinary sub-chunk allocation is VMM-backed, fully usable, and
// tears down cleanly under self-check.
TEST_F(MemoryManagerVMMTest, Phase0ChunkIsVmmBacked) {
  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, 4 * kGiB);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &freeList = *cache->freeList();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  ASSIGN_NO_ERR(void *, ptr,
                freeList.allocate(256 * kMiB, Device::AllocMode::SYNC));
  EXPECT_NE(ptr, nullptr);
  EXPECT_NO_ERR(writeAndVerify(*ctx, ptr, 256 * kMiB, 256 * kMiB));
  EXPECT_NO_ERR(freeList.free(ptr));
  cache->clear(); // standalone manager must be cleared before destruction
}

// Phase 1 (the fix): fragment the pool into wholly-free chunks whose largest
// contiguous block (1 GiB) is smaller than a 1.5 GiB request. With the pool
// full and no driver backdoor (memory-manager-only mode), the only way this
// request can succeed is VMM defrag assembling it from scattered free granules.
TEST_F(MemoryManagerVMMTest,
       DefragSatisfiesRequestExceedingLargestContiguousBlock) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 4 * kGiB;  // pool == 4 chunks of chunk_percent=25
  const size_t kChunk = 1 * kGiB;   // 1 GiB chunks (largest contiguous block)
  const size_t kBig = 3 * kGiB / 2; // 1.5 GiB > any single chunk

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  // Fill the pool with four 1 GiB buffers (one per chunk).
  std::vector<void *> bufs;
  for (int i = 0; i < 4; ++i) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kChunk, mode));
    bufs.push_back(p);
  }
  // Free two non-adjacent chunks: 2 GiB free, but largest contiguous is 1 GiB.
  EXPECT_NO_ERR(fl.free(bufs[0]));
  EXPECT_NO_ERR(fl.free(bufs[2]));

  // Defrag must assemble the 1.5 GiB request from the two scattered free
  // chunks.
  ASSIGN_NO_ERR(void *, big, fl.allocate(kBig, mode));
  EXPECT_NE(big, nullptr);
  // Prove the assembled buffer is fully backed and accessible across granules.
  EXPECT_NO_ERR(writeAndVerify(*ctx, big, kBig, 256 * kMiB));

  EXPECT_NO_ERR(fl.free(big));
  EXPECT_NO_ERR(fl.free(bufs[1]));
  EXPECT_NO_ERR(fl.free(bufs[3]));
  cache->clear();
}

// A freed block waits on a stream event before it can migrate to the parent, so
// the manager holds a record naming the freelist that owns it. Destroying that
// freelist must cancel the record: the manager outlives every freelist, so a
// surviving record would have it reach through a dangling pointer on the next
// allocation miss.
TEST_F(MemoryManagerVMMTest, PendingFreeCancelledWhenFreeListDestroyed) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, 4 * kGiB);
  ASSERT_NE(cache, nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  {
    // Deferral is on, so freeing through this child parks a record.
    MemoryManagerFreeList child(ctx.getPointer(), *cache, cache->freeList(),
                                /*disableFlush=*/false, /*label=*/"cancel");
    ASSIGN_NO_ERR(void *, ptr, child.allocate(256 * kMiB, mode));
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_ERR(child.free(ptr));
    // `child` is destroyed here with its record still queued.
  }

  // Would flush through freed memory if the record had outlived its freelist.
  EXPECT_NO_ERR(cache->drainCompletedFrees());

  // The destructor's own flush already returned the block, so the full budget
  // is allocatable again.
  MemoryManagerFreeList &root = *cache->freeList();
  ASSIGN_NO_ERR(void *, reused, root.allocate(256 * kMiB, mode));
  EXPECT_NE(reused, nullptr);
  EXPECT_NO_ERR(root.free(reused));
  cache->clear();
}

// An event recorded into a capturing stream becomes a graph node and reports
// ready before the graph has ever run. Deferring a free against one would hand
// the block back to the parent while the GPU may still read it, so no record is
// created during capture and the block stays put until a later free or
// teardown.
TEST_F(MemoryManagerVMMTest, NoDeferredFreeRecordedDuringGraphCapture) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, 4 * kGiB);
  ASSERT_NE(cache, nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  {
    MemoryManagerFreeList child(ctx.getPointer(), *cache, cache->freeList(),
                                /*disableFlush=*/false, /*label=*/"capture");
    ASSIGN_NO_ERR(void *, ptr, child.allocate(256 * kMiB, mode));
    EXPECT_NE(ptr, nullptr);

    // Capture on the context's own stream, which is what isGraphCapturing()
    // queries.
    ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
    EXPECT_NO_ERR(stream->beginCapture());
    ASSERT_TRUE(ctx->isGraphCapturing());
    EXPECT_NO_ERR(child.free(ptr));
    ErrorOr<DeviceGraphRef> graph = stream->endCapture();
    EXPECT_FALSE(graph.isError()) << graph.getError();

    // With a record, the drain would migrate the block to the parent and leave
    // this freelist empty.
    EXPECT_NO_ERR(cache->drainCompletedFrees());
    EXPECT_FALSE(child.empty());
  }
  cache->clear();
}

// A queued deferred free holds an event and a free-list pointer, never a
// DeviceContextRef. Were it to hold one, a context whose gating event cannot
// fire -- a trapped kernel, say -- would never be destroyed.
TEST_F(MemoryManagerVMMTest, PendingFreeDoesNotKeepContextAlive) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;

  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  MemoryManagerFreeList &fl = ctx->freeList();

  // Freeing rotates the ring and queues a record gated on a fresh event.
  ASSIGN_NO_ERR(void *, ptr, fl.allocate(4 * kMiB, mode));
  EXPECT_NO_ERR(fl.free(ptr));

  // This test's own reference must be the only one left.
  EXPECT_EQ(ctx->getNumReferences(), 1u);

  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
