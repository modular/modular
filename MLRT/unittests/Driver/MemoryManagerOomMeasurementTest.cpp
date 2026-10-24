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
// Measurement test for computeVmmOomSnapshot (GEX-3891). Rather than a failure
// condition, this builds one deliberately complex mixed state and checks every
// captured tier is exact -- verifying the granule/desc accounting handles
// sub-granule, multi-granule, mid-granule, and remainder regions in all three
// desc states (in-use / pending / root-free) at once.
//
// It drives the manager at the low level (a standalone makeCache root free-list
// for deterministic in-use and root-free descs -- root frees are root-owned
// immediately, no migration -- plus a child free-list for pending descs, which
// stay child-owned until an event fires). Everything is allocated up front so
// the layout is a deterministic bump from offset 0; then descs are freed into
// their target states with no re-allocation, so no reclamation churns the
// backing.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace M::Driver;
using M::ErrorOr;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;
constexpr size_t kPage = 2 * kMiB;    // matches MEMORY_MANAGER_VMM_PAGE_MB
constexpr size_t kBudget = 32 * kMiB; // 16 pages

class MemoryManagerOomMeasurementTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping measurement test";
#if MODULAR_MACOS
    GTEST_SKIP() << "VMM not supported on macOS, skipping measurement test";
#else
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping measurement test";
#endif
  }

  std::unique_ptr<MemoryManager> makeCache(Device &device, size_t budgetBytes) {
    ErrorOr<Device::MemInfo> memInfo = device.getMemoryInfo();
    if (memInfo.isError())
      return nullptr;
    return std::unique_ptr<MemoryManager>(
        MemoryManager::onDevice(device, *memInfo, device.allocationAlignment(),
                                static_cast<int64_t>(budgetBytes)));
  }
};

// A 32 MiB / 16-page budget laid out (bump from 0, sizes in MiB) as:
//
//   page0 [0,2)   : A0 in-use [0,1.5)   | A1 free  [1.5,2)     -> in-use page
//   page1 [2,4)   : A2 in-use [2,4)                            -> in-use page
//   page2 [4,6)   : A2 in-use [4,5) | A3 in-use [5,5.5) | A4 free [5.5,6)
//   page3 [6,8)   : B0 pend  [6,7.5)  | B1 free  [7.5,8)       -> pending page
//   page4 [8,10)  : B2 pend  [8,10)                            -> pending page
//   page5 [10,12) : B2 pend [10,11) | B3 pend [11,11.5) | B4 free [11.5,12)
//   page6 [12,14) : C0 free  [12,14)                          -> reclaimable
//   page7 [14,16) : C0 free [14,15) | C1 free [15,16)         -> reclaimable
//   pages 8..15   : never allocated                           -> uncommitted
//
// covering: in-use/pending/free that are sub-page (A0,A1,B0,B1,...), span
// multiple pages (A2,B2,C0), start mid-page (A3,A4,B3,B4,C1), and remainders
// sharing a page with an in-use desc (A1,A4) and with a pending desc (B1,B4).
TEST_F(MemoryManagerOomMeasurementTest, MixedStateSnapshotIsExact) {
  ASSIGN_NO_ERR(DeviceRef, device, Device::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  MemoryManagerFreeList &root = *cache->freeList();

  {
    // Child free-list (per-stream): its frees stay pending until an event
    // fires.
    MemoryManagerFreeList child(ctx.getPointer(), *cache, cache->freeList(),
                                /*disableFlush=*/false, "child");

    // Allocate everything up front, in offset order, from the two free-lists.
    ASSIGN_NO_ERR(void *, a0, root.allocate(kMiB * 3 / 2, mode));  // in-use
    ASSIGN_NO_ERR(void *, a1, root.allocate(kMiB / 2, mode));      // free
    ASSIGN_NO_ERR(void *, a2, root.allocate(kMiB * 3, mode));      // in-use
    ASSIGN_NO_ERR(void *, a3, root.allocate(kMiB / 2, mode));      // in-use
    ASSIGN_NO_ERR(void *, a4, root.allocate(kMiB / 2, mode));      // free
    ASSIGN_NO_ERR(void *, b0, child.allocate(kMiB * 3 / 2, mode)); // pending
    ASSIGN_NO_ERR(void *, b1, root.allocate(kMiB / 2, mode));      // free
    ASSIGN_NO_ERR(void *, b2, child.allocate(kMiB * 3, mode));     // pending
    ASSIGN_NO_ERR(void *, b3, child.allocate(kMiB / 2, mode));     // pending
    ASSIGN_NO_ERR(void *, b4, root.allocate(kMiB / 2, mode));      // free
    ASSIGN_NO_ERR(void *, c0, root.allocate(kMiB * 3, mode));      // free
    ASSIGN_NO_ERR(void *, c1, root.allocate(kMiB, mode));          // free

    // Root frees are root-owned immediately (no migration).
    EXPECT_NO_ERR(root.free(a1));
    EXPECT_NO_ERR(root.free(a4));
    EXPECT_NO_ERR(root.free(b1));
    EXPECT_NO_ERR(root.free(b4));
    EXPECT_NO_ERR(root.free(c0));
    EXPECT_NO_ERR(root.free(c1));
    // Child frees stay pending (no sync/poll yet).
    EXPECT_NO_ERR(child.free(b0));
    EXPECT_NO_ERR(child.free(b2));
    EXPECT_NO_ERR(child.free(b3));

    VmmOomSnapshot s = cache->oomSnapshotForTesting(0);

    // Page-state partition (16 pages).
    EXPECT_EQ(s.pageSize, kPage);
    EXPECT_EQ(s.inUseGranules, 3u);       // pages 0,1,2
    EXPECT_EQ(s.pendingGranules, 3u);     // pages 3,4,5
    EXPECT_EQ(s.reclaimableGranules, 2u); // pages 6,7
    EXPECT_EQ(s.pooledGranules, 0u);
    EXPECT_EQ(s.uncommittedGranules, 8u); // pages 8..15

    // Byte totals.
    EXPECT_EQ(s.usedBytes, 5 * kMiB);         // A0 + A2 + A3
    EXPECT_EQ(s.pendingTotalBytes, 5 * kMiB); // B0 + B2 + B3

    // Lever aggregates (and the conservation identity).
    EXPECT_EQ(s.available(), 20 * kMiB); // (2 + 0 + 8) pages
    EXPECT_EQ(s.recoverableBySync(),
              6 * kMiB);              // 3 pages (incl. B1,B4 remainders)
    EXPECT_EQ(s.trapped(), 1 * kMiB); // A1 + A4 in in-use pages
    EXPECT_EQ(s.usedBytes + s.available() + s.recoverableBySync() + s.trapped(),
              kBudget);

    // Teardown: free the in-use descs, then fire the child's pending frees so
    // they migrate to root before the child free-list is destroyed.
    EXPECT_NO_ERR(root.free(a0));
    EXPECT_NO_ERR(root.free(a2));
    EXPECT_NO_ERR(root.free(a3));
    EXPECT_NO_ERR(ctx->synchronize());
  }

  cache->clear();
}

// What a drain actually recovers, which the allocation path gates its retry on.
// Byte totals cannot answer that question in either direction, and this state
// exercises both: a page is handed back only when nothing in use sits on it and
// every child-owned desc on it is in a buffer some queued event names.
//
// Only the first free on a free-list queues an event; a later one, landing
// while that flush is still in flight, joins the rotated primary buffer that no
// record names. Three child free-lists therefore give three queued descs, and
// child1's second free gives a stuck one.
//
//   page0 [0,2)  : Q0 queued 0.5 | R0 root-free 1.5  -> recovered
//   page1 [2,4)  : Q1 queued 1.0 | S1 stuck     1.0  -> pinned by the straggler
//   page2 [4,6)  : U0 in-use 2.0                     -> in use
//   page3 [6,8)  : Q2 queued 1.0 | I1 in-use    1.0  -> pinned by the in-use
//
// So 2.5 MiB is drainable but only page0 (2 MiB) comes back, while the
// snapshot's recoverableBySync() counts pages 0 and 1 (4 MiB) because it cannot
// see that S1 has no event coming.
TEST_F(MemoryManagerOomMeasurementTest, DrainableRecoveryCountsWholeGranules) {
  ASSIGN_NO_ERR(DeviceRef, device, Device::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  MemoryManagerFreeList &root = *cache->freeList();

  {
    MemoryManagerFreeList child1(ctx.getPointer(), *cache, cache->freeList(),
                                 /*disableFlush=*/false, "child1");
    MemoryManagerFreeList child2(ctx.getPointer(), *cache, cache->freeList(),
                                 /*disableFlush=*/false, "child2");
    MemoryManagerFreeList child3(ctx.getPointer(), *cache, cache->freeList(),
                                 /*disableFlush=*/false, "child3");

    // Allocated in offset order so the bump layout matches the map above.
    ASSIGN_NO_ERR(void *, q0, child1.allocate(kMiB / 2, mode));
    ASSIGN_NO_ERR(void *, r0, root.allocate(kMiB * 3 / 2, mode));
    ASSIGN_NO_ERR(void *, q1, child2.allocate(kMiB, mode));
    ASSIGN_NO_ERR(void *, s1, child1.allocate(kMiB, mode));
    ASSIGN_NO_ERR(void *, u0, root.allocate(kMiB * 2, mode));
    ASSIGN_NO_ERR(void *, q2, child3.allocate(kMiB, mode));
    ASSIGN_NO_ERR(void *, i1, root.allocate(kMiB, mode));

    // First free on each child queues an event naming the buffer it lands in.
    EXPECT_NO_ERR(child1.free(q0));
    // Second free on child1: its flush is already in flight, so this one joins
    // the rotated primary and no queued record names it.
    EXPECT_NO_ERR(child1.free(s1));
    EXPECT_NO_ERR(child2.free(q1));
    EXPECT_NO_ERR(child3.free(q2));
    EXPECT_NO_ERR(root.free(r0));

    const VmmOomSnapshot s = cache->oomSnapshotForTesting(0);
    EXPECT_EQ(s.drainableBytes, kMiB * 5 / 2); // Q0 + Q1 + Q2
    EXPECT_EQ(s.drainableGranules, 1u);        // page0 only

    // The point of counting granules rather than bytes: the drainable bytes
    // exceed what a drain hands back, while on page0 half a MiB releases a
    // whole 2 MiB page. Either figure alone misleads, in opposite directions.
    EXPECT_GT(s.drainableBytes, s.drainableGranules * kPage);

    // And the older lever over-states it, because S1 is pending with no event
    // coming. Both now live in the same snapshot, so a reader of the OOM dump
    // sees the gap between them.
    EXPECT_EQ(s.recoverableBySync(), 2 * kPage);
    EXPECT_LT(s.drainableGranules * kPage, s.recoverableBySync());

    EXPECT_NO_ERR(root.free(u0));
    EXPECT_NO_ERR(root.free(i1));
    EXPECT_NO_ERR(ctx->synchronize());
  }

  cache->clear();
}

} // namespace
