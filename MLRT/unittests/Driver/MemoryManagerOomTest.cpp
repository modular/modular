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
// Device-context-level OOM tests for the single-arena VMM allocator (GEX-3890).
// Each test drives a real DeviceContext through the public buffer API
// (`enqueueCreateBuffer`) and deterministically produces an out-of-memory error
// for one specific reason:
//
//   1. Genuine    - the physical budget is fully committed by live allocations.
//   2. Pending    - enough memory exists, but it is held in a sibling stream's
//                   per-stream free list behind an unfired event; a sync frees
//                   it.
//   3. Fragmented - enough memory exists, but it is stranded in partially-used
//                   granules, so no whole granule can back a new allocation.
//
// A fourth test covers the reclaimable page state, which none of the three
// failures above produce.
//
// Conditions 2 and 3 both surface as `recoverable`: whether a synchronize or a
// smaller page is the cheaper fix depends on the application, so the category
// says only that this is not an exhaustion and the message reports what each
// lever would recover. The tests tell them apart by the lever named and by the
// snapshot tier values, which is where they genuinely differ.
//
// Each test asserts both the miss reason -- the parenthesised tag, which is the
// one place a category appears unambiguously -- and the exact snapshot tier
// values the classification rests on.
//
// The backdoor is disabled (memory-manager-only mode) so a miss surfaces as an
// error rather than falling through to the driver; MemoryManagerOomBackdoor-
// Test covers the fallthrough. The physical budget is capped via
// MEMORY_MANAGER_SIZE (see BUILD) so these run cheaply. Config comes from the
// target env (parsed once per process). The side-channel case is a separate,
// resource-heavy target (MemoryManagerOomSideChannelTest.cpp).
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;

// Physical budget and page size for this target. Must match
// MEMORY_MANAGER_SIZE and MEMORY_MANAGER_VMM_PAGE_MB in the BUILD env; the
// deliberate allocation counts and snapshot assertions below are derived from
// them, so a drift surfaces as a failed assertion rather than a silent
// miscalibration.
constexpr size_t kBudget = 256 * kMiB;
constexpr size_t kPage = 32 * kMiB;

// The device's memory manager (the one enqueueCreateBuffer allocates from).
MemoryManager &mm(DeviceContext &ctx) { return ctx.device().memoryManager(); }

class MemoryManagerOomTest : public ::testing::Test {
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
#endif
  }
};

// Condition 1: genuine OOM. Allocate exactly enough buffers to commit the whole
// budget, keeping them all live; the next allocation then has no pool handle,
// no budget headroom, and nothing reclaimable, so it must fail.
TEST_F(MemoryManagerOomTest, GenuineOutOfMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  const size_t bufSize = 16 * kMiB;
  const size_t count = kBudget / bufSize; // fills the budget exactly

  std::vector<DeviceBufferRef> live;
  for (size_t i = 0; i < count; ++i) {
    ASSIGN_NO_ERR(DeviceBufferRef, b, ctx->enqueueCreateBuffer(bufSize));
    live.push_back(std::move(b));
  }

  // The budget is now fully committed: one more allocation must fail.
  ErrorOr<DeviceBufferRef> more = ctx->enqueueCreateBuffer(bufSize);
  EXPECT_TRUE(more.isError()) << "expected genuine OOM once the budget is full";
  EXPECT_THAT(more.getError(), ::testing::HasSubstr("(out of memory)"));

  // Nothing is recoverable: the whole budget is genuinely in use.
  VmmOomSnapshot snap = mm(*ctx).oomSnapshotForTesting(bufSize);
  EXPECT_EQ(snap.available(), 0u);
  EXPECT_EQ(snap.recoverableBySync(), 0u);
  EXPECT_EQ(snap.trapped(), 0u);
  EXPECT_EQ(snap.usedBytes, kBudget);
  EXPECT_EQ(snap.inUseGranules, kBudget / kPage);

  live.clear();
  EXPECT_NO_ERR(ctx->synchronize());
}

// Condition 2: pending memory. A holder context allocates one buffer covering
// most of the budget and frees it behind a host-controlled gate, so the freed
// physical stays child-owned (unreclaimable) and never migrates to the root. A
// second context therefore cannot get it and OOMs. Once the gate is signalled
// and the holder synchronized, the memory migrates and the same request
// succeeds -- proving the shortfall was pending, not genuine.
TEST_F(MemoryManagerOomTest, PendingMemoryBlocksUntilSync) {
  // One buffer dominating the budget: freeing a single desc enqueues exactly
  // one flush-to-root handler, so the whole amount migrates on sync. The
  // request equals this size, and the budget headroom left after the holder
  // frees (< 64 MiB) cannot satisfy it.
  const size_t bufSize = 192 * kMiB;

  ASSIGN_NO_ERR(DeviceContextRef, holder,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, allocator,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceStreamRef, holderStream, holder->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, gate, holder->allocateCompletionFlag());
  gate->reset();

  ASSIGN_NO_ERR(DeviceBufferRef, held, holder->enqueueCreateBuffer(bufSize));

  // Close the gate, THEN free: the free's flush-to-root event is ordered after
  // the gate and cannot fire, so the memory stays pending in the holder's list.
  EXPECT_NO_ERR(holderStream->enqueueWaitOnHostValue(*gate, 1));
  held.reset();

  // The allocator cannot reach the holder's gated (pending) physical. The miss
  // path exhausts its drain retries first and gets nothing, since the gate only
  // opens below -- which is also what proves that step cannot deadlock on an
  // event the calling thread itself is holding shut. Costs those retries in
  // wall time.
  ErrorOr<DeviceBufferRef> blocked = allocator->enqueueCreateBuffer(bufSize);
  EXPECT_TRUE(blocked.isError())
      << "expected OOM while freed memory is pending behind the gate";
  EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("(avoidable)"));
  EXPECT_THAT(blocked.getError(), ::testing::HasSubstr("synchroniz"));

  // The shortfall is entirely pending: a sync would recover the held buffer.
  // (Assert the lever aggregates, which are stable across the failed attempt's
  // rollback; the individual uncommitted/pooled split is not.)
  VmmOomSnapshot snap = mm(*allocator).oomSnapshotForTesting(bufSize);
  EXPECT_EQ(snap.recoverableBySync(), bufSize);
  EXPECT_EQ(snap.pendingGranules, bufSize / kPage);
  EXPECT_EQ(snap.usedBytes, 0u);
  EXPECT_EQ(snap.trapped(), 0u);
  EXPECT_LT(snap.available(), bufSize);

  // Open the gate and drain the holder's stream so its frees migrate to root.
  gate->signal(1);
  EXPECT_NO_ERR(holder->synchronize());

  // The same request now succeeds purely because we synchronized.
  ErrorOr<DeviceBufferRef> ok = allocator->enqueueCreateBuffer(bufSize);
  if (ok.isError()) {
    ADD_FAILURE() << "expected the request to succeed after sync: "
                  << ok.getError();
  } else {
    // Free before the trailing sync: a buffer freed at context-destruction time
    // with no synchronize after leaves a pending flush handler that keeps the
    // context alive and registered, which a later allocation would deadlock on.
    ok.takeValue().reset();
  }
  EXPECT_NO_ERR(allocator->synchronize());
}

// The allocator's own recovery from condition 2. Same setup as above, except
// the gate opens while the request is in flight: the miss path classifies the
// shortfall as avoidable and retries on its pending frees, so the request is
// served by memory that had not yet migrated when it started. No caller-side
// synchronize, and no OOM. Before that step this request failed outright,
// taking the process with it under memory-manager-only mode.
TEST_F(MemoryManagerOomTest, PendingMemoryRecoveredWithoutCallerSync) {
  const size_t bufSize = 192 * kMiB;

  ASSIGN_NO_ERR(DeviceContextRef, holder,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, allocator,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceStreamRef, holderStream, holder->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, gate, holder->allocateCompletionFlag());
  gate->reset();

  ASSIGN_NO_ERR(DeviceBufferRef, held, holder->enqueueCreateBuffer(bufSize));
  EXPECT_NO_ERR(holderStream->enqueueWaitOnHostValue(*gate, 1));
  held.reset();

  // Counters live on the device's manager, which the whole suite shares.
  const size_t attemptsBefore = mm(*allocator).drainRetryAttempts();
  const size_t recoveriesBefore = mm(*allocator).drainRetryRecoveries();

  // Open the gate only once the request has entered the retry path. A fixed
  // sleep would race: were it to elapse first, the existing poll would serve
  // the request, the counters would not move, and the assertions below would
  // fail. `signal` is a release store to pinned host memory, so driving it
  // from another thread is sound.
  std::atomic<bool> handshook{false};
  std::thread opener([&] {
    // Bounded so a request that never retries releases this thread instead of
    // hanging the join. That exit is a failure, not a handshake, and is
    // reported below rather than being papered over by the signal.
    for (int i = 0; i < 2000; ++i) {
      if (mm(*allocator).drainRetryAttempts() != attemptsBefore) {
        handshook.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    gate->signal(1);
  });

  ErrorOr<DeviceBufferRef> recovered = allocator->enqueueCreateBuffer(bufSize);
  opener.join();

  EXPECT_TRUE(handshook.load())
      << "the request never entered the retry path, so this run proves nothing "
         "about it";
  if (recovered.isError()) {
    ADD_FAILURE() << "expected the drain retry to serve this: "
                  << recovered.getError();
  } else {
    // The request went through the new path rather than being served outright.
    EXPECT_EQ(mm(*allocator).drainRetryAttempts(), attemptsBefore + 1);
    EXPECT_EQ(mm(*allocator).drainRetryRecoveries(), recoveriesBefore + 1);
    recovered.takeValue().reset();
  }
  EXPECT_NO_ERR(holder->synchronize());
  EXPECT_NO_ERR(allocator->synchronize());
}

// The operator's off switch. Same pending shortfall as the two tests above,
// but the target sets memory_manager_vmm_drain_retries=0, so the miss must
// fail the way it did before the retry existed: no attempt recorded, and no
// time spent waiting for a free that would have arrived.
TEST_F(MemoryManagerOomTest, DrainRetriesZeroDisablesTheRetry) {
  const size_t bufSize = 192 * kMiB;

  ASSIGN_NO_ERR(DeviceContextRef, holder,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, allocator,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceStreamRef, holderStream, holder->stream());
  ASSIGN_NO_ERR(CompletionFlagRef, gate, holder->allocateCompletionFlag());
  gate->reset();

  ASSIGN_NO_ERR(DeviceBufferRef, held, holder->enqueueCreateBuffer(bufSize));
  EXPECT_NO_ERR(holderStream->enqueueWaitOnHostValue(*gate, 1));
  held.reset();

  const size_t attemptsBefore = mm(*allocator).drainRetryAttempts();
  ErrorOr<DeviceBufferRef> blocked = allocator->enqueueCreateBuffer(bufSize);
  EXPECT_TRUE(blocked.isError())
      << "expected OOM while freed memory is pending behind the gate";
  EXPECT_EQ(mm(*allocator).drainRetryAttempts(), attemptsBefore)
      << "the retry ran despite being configured off";

  gate->signal(1);
  EXPECT_NO_ERR(holder->synchronize());
  EXPECT_NO_ERR(allocator->synchronize());
}

// Condition 3: fragmentation. Allocate exactly two half-granule buffers per
// granule to fill the budget, then free every other one. Every granule keeps
// one live half, so no whole granule is free even though half the budget's
// bytes are now unused. A half-granule request still succeeds (the freed halves
// are backed and reusable), but a full-granule request cannot be satisfied: no
// whole granule is free and there is no budget headroom to create new physical.
TEST_F(MemoryManagerOomTest, FragmentedPartialGranulesBlockWholeGranule) {
  // Matches MEMORY_MANAGER_VMM_PAGE_MB in the BUILD target. A large granule
  // keeps the buffer count small (the public buffer API has meaningful
  // per-allocation overhead) while still forcing the partial-granule pattern.
  const size_t granule = 32 * kMiB;
  const size_t half = granule / 2;
  const size_t count = kBudget / half; // two halves per granule, fills budget

  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  std::vector<DeviceBufferRef> halves;
  for (size_t i = 0; i < count; ++i) {
    ASSIGN_NO_ERR(DeviceBufferRef, b, ctx->enqueueCreateBuffer(half));
    halves.push_back(std::move(b));
  }

  // Free every other buffer: each granule keeps one live half, so every granule
  // stays pinned while half the budget's bytes become free-but-fragmented.
  for (size_t i = 1; i < count; i += 2)
    halves[i].reset();
  EXPECT_NO_ERR(ctx->synchronize());

  // A whole-granule request cannot be satisfied despite ample free bytes.
  ErrorOr<DeviceBufferRef> whole = ctx->enqueueCreateBuffer(granule);
  EXPECT_TRUE(whole.isError())
      << "expected OOM: free memory is stranded in partially-used granules";
  EXPECT_THAT(whole.getError(), ::testing::HasSubstr("(avoidable)"));
  EXPECT_THAT(whole.getError(),
              ::testing::HasSubstr("memory_manager_vmm_page_mb"));
  // The figure matters as much as the lever: half the budget (kBudget / 2) is
  // trapped here, and a message that named the lever without the amount would
  // not tell the reader whether it was worth pulling.
  EXPECT_THAT(whole.getError(), ::testing::HasSubstr("128MB"));

  // Half the budget is trapped in partially-used granules: nothing allocatable
  // now or via sync, but a smaller page would recover it.
  VmmOomSnapshot snap = mm(*ctx).oomSnapshotForTesting(granule);
  EXPECT_EQ(snap.available(), 0u);
  EXPECT_EQ(snap.recoverableBySync(), 0u);
  EXPECT_EQ(snap.trapped(), kBudget / 2);
  EXPECT_EQ(snap.usedBytes, kBudget / 2);
  EXPECT_EQ(snap.inUseGranules, kBudget / kPage);

  // A half-granule request still succeeds, proving the memory exists and the
  // failure was fragmentation, not genuine exhaustion.
  ErrorOr<DeviceBufferRef> reuse = ctx->enqueueCreateBuffer(half);
  EXPECT_FALSE(reuse.isError())
      << "expected a half-granule allocation to reuse freed space";

  halves.clear();
  EXPECT_NO_ERR(ctx->synchronize());
}

// Coverage for the reclaimable tier, which the OOM conditions above never
// produce (they leave it at 0). Free one whole-granule buffer and synchronize
// so its single desc migrates to the root still backed; the snapshot must then
// count it as a reclaimable granule (backed, root-owned, no in-use/pending).
TEST_F(MemoryManagerOomTest, SnapshotCountsReclaimableGranule) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  ASSIGN_NO_ERR(DeviceBufferRef, keep, ctx->enqueueCreateBuffer(kPage));
  ASSIGN_NO_ERR(DeviceBufferRef, freed, ctx->enqueueCreateBuffer(kPage));
  freed.reset(); // single free -> one flush handler -> fully migrates on sync
  EXPECT_NO_ERR(ctx->synchronize());

  VmmOomSnapshot snap = mm(*ctx).oomSnapshotForTesting(kPage);
  EXPECT_GE(snap.reclaimableGranules, 1u)
      << "a freed, synced, still-backed whole granule should be reclaimable";
  EXPECT_EQ(snap.inUseGranules, 1u); // `keep`
  EXPECT_GE(snap.available(), kPage);

  keep.reset();
  EXPECT_NO_ERR(ctx->synchronize());
}

} // namespace
