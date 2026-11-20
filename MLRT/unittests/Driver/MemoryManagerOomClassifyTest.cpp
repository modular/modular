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
// GPU-free unit tests for the VMM OOM classifier and formatters (GEX-3890).
// These exercise `classifyVmmOom` / `describeVmmOom` / `dumpVmmOom` over
// hand-built `VmmOomSnapshot` values, so every failure category -- including
// the ones that are hard to reproduce on real hardware (unaccounted and
// VA-fragmentation) and the pending-trapped-in-in-use counterexample -- is
// covered deterministically without a device.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"

#include "gtest/gtest.h"

#include <cstddef>

using namespace M::Driver;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;
constexpr size_t kPage = 2 * kMiB;      // granule
constexpr size_t kBudgetGranules = 100; // 200 MiB budget

// A snapshot with a 2 MiB page and a 200 MiB budget over a reserved arena;
// callers set the granule counts / byte totals to shape the scenario.
VmmOomSnapshot base(size_t needed) {
  VmmOomSnapshot s;
  s.pageSize = kPage;
  s.budget = kBudgetGranules * kPage;
  s.needed = needed;
  s.arenaReserved = true;
  return s;
}

// The drain figures are the only inputs that vary across the retry cases, so
// set them on a copy rather than restating the whole snapshot each time.
VmmOomSnapshot withDrainableSnapshot(VmmOomSnapshot s, size_t bytes,
                                     size_t granules) {
  s.drainableBytes = bytes;
  s.drainableGranules = granules;
  return s;
}

bool withDrainable(const VmmOomSnapshot &s, size_t bytes, size_t granules) {
  return pendingRetryWorthwhile(withDrainableSnapshot(s, bytes, granules));
}

TEST(MemoryManagerOomClassify, Genuine) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.inUseGranules = kBudgetGranules; // whole budget pinned by live allocations
  s.usedBytes = s.budget;
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Genuine);
  EXPECT_NE(describeVmmOom(s, VmmOomCategory::Genuine).find("out of memory"),
            std::string::npos);
}

// Genuine fires whenever no lever closes the gap, which includes states where
// some memory is still allocatable. The message must not claim the budget is
// fully committed, since it prints the allocatable figure alongside.
TEST(MemoryManagerOomClassify, GenuineWithPartialAvailabilityIsNotOverclaimed) {
  VmmOomSnapshot s = base(60 * kMiB);
  s.reclaimableGranules = 5; // available = 10 MiB, short of the 60 MiB need
  s.inUseGranules = 95;
  s.usedBytes = s.budget - 10 * kMiB;
  ASSERT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Genuine);
  std::string msg = describeVmmOom(s, VmmOomCategory::Genuine);
  // The point is the claim, not the byte formatting: some memory is
  // allocatable, so the message must not say the budget is committed.
  EXPECT_NE(msg.find("allocatable"), std::string::npos);
  EXPECT_NE(msg.find("no combination"), std::string::npos);
  EXPECT_EQ(msg.find("is committed"), std::string::npos);
}

TEST(MemoryManagerOomClassify, SyncRecoverableOnly) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.pendingGranules = 20;    // sync-recoverable = 40 MiB
  s.inUseGranules = 78;
  // The in-use pages are fully occupied, so nothing is trapped. Without this
  // the pages would be pinned while holding no bytes, which cannot happen: the
  // whole of each page would count as trapped.
  s.usedBytes = 78 * kPage;
  s.pendingTotalBytes = 40 * kMiB;
  // available (4) < need (20); available + sync (44) >= need.
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Avoidable);
  ASSERT_EQ(s.trapped(), 0u);
  // Nothing is trapped, so the message names only the synchronize lever.
  std::string msg = describeVmmOom(s, VmmOomCategory::Avoidable);
  EXPECT_NE(msg.find("synchroniz"), std::string::npos);
  EXPECT_EQ(msg.find("partially-used"), std::string::npos);
}

TEST(MemoryManagerOomClassify, TrappedOnly) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.inUseGranules = 98;      // pinned granules
  s.usedBytes = 100 * kMiB;  // trapped = 98*2 - 100 = 96 MiB
  // available + sync (4) < need; available + trapped (100) >= need.
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Avoidable);
  // Nothing is pending, so the message names only the page-size lever.
  std::string msg = describeVmmOom(s, VmmOomCategory::Avoidable);
  EXPECT_NE(msg.find("memory_manager_vmm_page_mb"), std::string::npos);
  EXPECT_EQ(msg.find("synchroniz"), std::string::npos);
}

// The key correctness case: a large amount of pending memory that is
// co-resident with in-use allocations. Those granules are classified in-use, so
// pendingGranules (and thus sync-recoverable) is zero -- synchronizing would
// free the descs but recover no allocatable granule. Must NOT be Pending.
TEST(MemoryManagerOomClassify, PendingTrappedWithInUseIsNotPending) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.pendingGranules = 0;     // sync-recoverable = 0 despite pending bytes
  s.pendingTotalBytes = 100 * kMiB; // lots pending, all sharing in-use granules
  s.inUseGranules = 98;
  s.usedBytes = 100 * kMiB; // trapped = 96 MiB
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Avoidable);
  // 100 MiB is pending, but none of it is sync-recoverable, so offering a
  // synchronize here would send the reader after memory a sync cannot free.
  ASSERT_EQ(s.recoverableBySync(), 0u);
  std::string msg = describeVmmOom(s, c);
  EXPECT_EQ(msg.find("synchroniz"), std::string::npos);
  EXPECT_NE(msg.find("memory_manager_vmm_page_mb"), std::string::npos);
}

// When neither lever alone closes the gap but both together do, this is still
// recoverable rather than an exhaustion.
TEST(MemoryManagerOomClassify, BothLeversNeededIsStillRecoverable) {
  VmmOomSnapshot s = base(100 * kMiB);
  s.reclaimableGranules = 10; // available = 20 MiB
  s.pendingGranules = 10;     // recoverable_by_sync = 20 MiB
  s.inUseGranules = 80;
  s.usedBytes = 100 * kMiB; // trapped = 80*2 - 100 = 60 MiB
  // available + sync = 40 < 100 (sync alone insufficient);
  // available + trapped = 80 < 100 (page alone insufficient);
  // available + sync + trapped = 100 >= 100 (both together suffice).
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Avoidable);
  // Both levers carry memory, so both are named with their figures.
  std::string msg = describeVmmOom(s, VmmOomCategory::Avoidable);
  EXPECT_NE(msg.find("synchroniz"), std::string::npos);
  EXPECT_NE(msg.find("memory_manager_vmm_page_mb"), std::string::npos);
}

// The case that motivated merging the two: a small synchronize would satisfy
// the request, while far more memory sits trapped. Reporting only the
// synchronize would hide that fragmentation is the larger problem, and
// reporting only the trapped memory would recommend a global page-size change
// where a synchronize sufficed. Both figures are reported.
TEST(MemoryManagerOomClassify, SmallSyncBesideLargeTrappedReportsBoth) {
  VmmOomSnapshot s = base(50 * kMiB);
  s.reclaimableGranules = 20; // available = 40 MiB
  s.pendingGranules = 5;      // sync-recoverable = 10 MiB, alone enough
  s.pendingTotalBytes = 10 * kMiB;
  s.inUseGranules = 75;
  s.usedBytes = 90 * kMiB; // trapped = 75*2 - 90 = 60 MiB, far larger
  ASSERT_GE(s.available() + s.recoverableBySync(), 50 * kMiB);
  ASSERT_GT(s.trapped(), s.recoverableBySync());
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Avoidable);
  std::string msg = describeVmmOom(s, c);
  EXPECT_NE(msg.find("synchroniz"), std::string::npos);
  EXPECT_NE(msg.find("partially-used"), std::string::npos);
  EXPECT_NE(msg.find("Not an exhaustion"), std::string::npos);
}

// DRIV-690's shape, scaled to this fixture: every pending granule is held by a
// graph-capture free list, which never migrates to its parent. Those bytes
// used to count as sync-recoverable, so the verdict read `avoidable` and the
// message offered a synchronize -- against ~50 GB no wait could ever free.
TEST(MemoryManagerOomClassify, RetainedIsNotSyncRecoverable) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.pendingGranules = 20;
  s.retainedGranules = 20; // all of it, so nothing is sync-recoverable
  s.inUseGranules = 78;
  s.usedBytes = 78 * kPage; // fully occupied, so nothing is trapped
  s.pendingTotalBytes = 40 * kMiB;
  s.retainedBytes = 40 * kMiB;
  s.pendingOwners = {{"graphFreeList", 40 * kMiB, 46, /*neverFlushes=*/true}};
  ASSERT_EQ(s.recoverableBySync(), 0u);
  ASSERT_EQ(s.retained(), 40 * kMiB);
  ASSERT_EQ(s.trapped(), 0u);

  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Retained);
  std::string msg = describeVmmOom(s, c);
  // The holder is the whole point: the free list the error names is the one
  // the allocation missed on, which is routinely not the one holding the
  // budget.
  EXPECT_NE(msg.find("graphFreeList"), std::string::npos);
  EXPECT_NE(msg.find("No synchronize recovers"), std::string::npos);
  // The offer, specifically. "No synchronize recovers that memory" above also
  // matches a bare search for the word.
  EXPECT_EQ(msg.find("that synchronizing would free"), std::string::npos);
}

// A retained block can share a page with a live allocation, which keeps the
// page pinned and so keeps the block out of `retainedGranules`. Those bytes
// used to fall into `trapped()`, which carries the page-size lever -- and a
// smaller page hands them back to their holder, not to the allocator. Neither
// lever frees them alone, so neither may claim them.
TEST(MemoryManagerOomClassify, RetainedOnLiveGranulesIsNotTrapped) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.inUseGranules = 98;
  s.usedBytes = 100 * kMiB; // 96 MiB of the live pages holds no live desc
  s.pendingTotalBytes = 90 * kMiB;
  s.retainedBytes = 90 * kMiB;
  s.retainedTrappedBytes = 90 * kMiB; // all of it beside a live allocation
  s.pendingOwners = {{"graphFreeList", 90 * kMiB, 30, /*neverFlushes=*/true}};
  // Nothing is wholly pending, so releasing the graphs frees no whole page.
  ASSERT_EQ(s.retained(), 0u);
  EXPECT_EQ(s.trapped(), 6 * kMiB); // 96 - 90, not 96

  // 4 + 0 + 6 = 10 MiB against a 20 MiB request, and the retained granule
  // figure adds nothing, so no lever closes this.
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Genuine);
  // Genuine still names the pool, since 90 MiB of the budget sits there.
  EXPECT_NE(describeVmmOom(s, c).find("graphFreeList"), std::string::npos);
}

// Retained bytes do not erase the ordinary ones beside them. When a wait does
// close the gap, the wait stays the headline and the retained pool is a note.
TEST(MemoryManagerOomClassify, RetainedBesideSyncRecoverableKeepsWaitAdvice) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.pendingGranules = 20;
  s.retainedGranules = 10; // half held, half waiting on stream frees
  s.inUseGranules = 78;
  s.usedBytes = 78 * kPage;
  s.pendingTotalBytes = 40 * kMiB;
  s.retainedBytes = 20 * kMiB;
  s.pendingOwners = {{"graphFreeList", 20 * kMiB, 8, /*neverFlushes=*/true},
                     {"freeList", 20 * kMiB, 4, /*neverFlushes=*/false}};
  ASSERT_EQ(s.recoverableBySync(), 20 * kMiB);
  // available + sync = 24 MiB >= the 20 MiB request, so a wait suffices.
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Avoidable);
  std::string msg = describeVmmOom(s, c);
  EXPECT_NE(msg.find("that synchronizing would free"), std::string::npos);
  EXPECT_NE(msg.find("graphFreeList"), std::string::npos);
  // The waiting-in-total figure excludes the retained half, which no wait
  // reaches; reporting the raw pending total would overstate what to expect.
  EXPECT_NE(msg.find("20MB waiting on stream frees in total"),
            std::string::npos);
}

// Retained memory too small to cover the request leaves the verdict a genuine
// exhaustion, but it is still the reason the budget ran out. Silence here
// reads as "the model needs this much memory", which sends the reader to the
// batch size rather than to the capture count.
TEST(MemoryManagerOomClassify, GenuineStillNamesTheRetainedPool) {
  VmmOomSnapshot s = base(100 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.pendingGranules = 20;
  s.retainedGranules = 20;
  s.inUseGranules = 78;
  s.usedBytes = 78 * kPage;
  s.pendingTotalBytes = 40 * kMiB;
  s.retainedBytes = 40 * kMiB;
  s.pendingOwners = {{"graphFreeList", 40 * kMiB, 46, /*neverFlushes=*/true}};
  // 4 + 0 sync + 0 trapped + 40 retained = 44 MiB, short of the 100 MiB need.
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Genuine);
  std::string msg = describeVmmOom(s, c);
  EXPECT_NE(msg.find("out of memory"), std::string::npos);
  EXPECT_NE(msg.find("graphFreeList"), std::string::npos);
}

// DRIV-690's own numbers, from the engine log of Mammoth run 41927518031: a
// MiniMax-M3.1 DSpark EP8/DP2 engine on 8x B200. The whole 54.03 GB pending
// total was 46 free blocks in one `graphFreeList`, left there by 222 decode
// shape captures. The engine missed twice, four minutes apart, and the fix
// has to hold for both.
class MemoryManagerDriv690 : public ::testing::Test {
protected:
  static constexpr size_t kGranule = 32 * kMiB;

  // 32 MiB pages over a 169.44 GiB budget, carrying the incident's granule
  // counts verbatim. Request sizes are rounded to whole MiB; nothing here
  // turns on the fraction.
  static VmmOomSnapshot incident(size_t needed, size_t reclaimableGranules) {
    VmmOomSnapshot s;
    s.pageSize = kGranule;
    s.needed = needed;
    s.arenaReserved = true;
    s.inUseGranules = 3723;
    s.pendingGranules = 1691;
    s.retainedGranules = 1691; // every one of them, held by graphFreeList
    s.reclaimableGranules = reclaimableGranules;
    s.budget =
        (s.inUseGranules + s.pendingGranules + reclaimableGranules) * kGranule;
    // 1250 MiB of the live granules holds no live desc, which the log reports
    // as 1.22GB trapped.
    s.usedBytes = s.inUseGranules * kGranule - 1250 * kMiB;
    s.pendingTotalBytes = 55327 * kMiB;
    s.retainedBytes = s.pendingTotalBytes;
    // Every pending desc sat in graphFreeList, so retained bytes that are not
    // on a wholly-pending granule must be on a live one. That is 1215 MiB of
    // the 1250 MiB above: nearly all of what the log called trapped was
    // really held by the capture pool.
    s.retainedTrappedBytes = s.retainedBytes - s.retainedGranules * kGranule;
    s.largestRootFreeVA = 508 * 1024 * kMiB;
    s.vaSize = 4 * s.budget;
    s.freeBlocks = 106;
    s.pendingOwners = {
        {"graphFreeList", s.retainedBytes, 46, /*neverFlushes=*/true}};
    return s;
  }
};

// The first miss: a 2.73 GB capture allocation during startup. No lever that
// works reaches it, so the pool that does hold the memory is the verdict.
TEST_F(MemoryManagerDriv690, CaptureMissIsRetained) {
  VmmOomSnapshot s = incident(2796 * kMiB, /*reclaimableGranules=*/17);
  ASSERT_EQ(s.available(), 544 * kMiB);
  ASSERT_EQ(s.recoverableBySync(), 0u);
  ASSERT_LT(s.available() + s.trapped(), s.needed);

  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Retained);
  std::string msg = describeVmmOom(s, c);
  EXPECT_NE(msg.find("graphFreeList"), std::string::npos);
  EXPECT_NE(msg.find("Capture fewer shapes"), std::string::npos);
  EXPECT_EQ(msg.find("that synchronizing would free"), std::string::npos);
}

// The trapped figure the log reported was almost entirely capture-pool memory
// sharing a page with a live allocation. Offering the page-size lever for it
// is the same error as offering a synchronize, so `trapped()` has to shed it.
TEST_F(MemoryManagerDriv690, TrappedShedsTheCapturePoolShare) {
  VmmOomSnapshot s = incident(266 * kMiB, /*reclaimableGranules=*/8);
  // What the raw difference says, and what the log printed.
  ASSERT_EQ(s.inUseGranules * kGranule - s.usedBytes, 1250 * kMiB);
  ASSERT_EQ(s.retainedTrappedBytes, 1215 * kMiB);
  // What a smaller page would really hand back.
  EXPECT_EQ(s.trapped(), 35 * kMiB);
}

// The first of the two fatal forward-pass misses: 265.50 MB. The corrected
// trapped figure still covers it next to the 256MB allocatable, so
// `avoidable` stands. It stands on 35MB rather than on 1.22GB, and the
// message has to name where the 52.84 GB it competed with sits.
TEST_F(MemoryManagerDriv690, SmallForwardPassMissStaysAvoidable) {
  VmmOomSnapshot s = incident(266 * kMiB, /*reclaimableGranules=*/8);
  ASSERT_EQ(s.available(), 256 * kMiB);
  ASSERT_EQ(s.recoverableBySync(), 0u);
  ASSERT_GE(s.available() + s.trapped(), s.needed);

  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Avoidable);
  std::string msg = describeVmmOom(s, c);
  // The page-size lever is the one that serves this request.
  EXPECT_NE(msg.find("memory_manager_vmm_page_mb"), std::string::npos);
  // Before the fix this read "up to 52.84GB that synchronizing would free",
  // which is what sent the ticket after the deferred-free drain.
  EXPECT_EQ(msg.find("synchronizing would free"), std::string::npos);
  EXPECT_NE(msg.find("graphFreeList"), std::string::npos);
  // And the dump carries the holder for whoever reads the log rather than the
  // message.
  EXPECT_NE(dumpVmmOom(s).find("graphFreeList=54.03GB/46(retained)"),
            std::string::npos);
}

// The second one, 480 MB with 160MB allocatable, logged in the same
// millisecond. This is where shedding the capture-pool share changes the
// verdict: on the old trapped figure it read `avoidable` and sent the reader
// to the page size, which would not have served it.
TEST_F(MemoryManagerDriv690, LargeForwardPassMissTurnsRetained) {
  VmmOomSnapshot s = incident(480 * kMiB, /*reclaimableGranules=*/5);
  ASSERT_EQ(s.available(), 160 * kMiB);
  // The uncorrected figure would have closed the gap on its own.
  ASSERT_GE(s.available() + 1250 * kMiB, s.needed);
  ASSERT_LT(s.available() + s.trapped(), s.needed);

  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::Retained);
  std::string msg = describeVmmOom(s, c);
  EXPECT_EQ(msg.find("memory_manager_vmm_page_mb"), std::string::npos);
  EXPECT_NE(msg.find("graphFreeList"), std::string::npos);
}

TEST(MemoryManagerOomClassify, Unaccounted) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.uncommittedGranules = 50; // available = 100 MiB: budget had headroom
  // A driver-side create failure overrides the accounting: the manager thought
  // it had room but the device refused.
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::CreateRefused);
  EXPECT_EQ(c, VmmOomCategory::Unaccounted);
  std::string msg = describeVmmOom(s, c);
  EXPECT_NE(msg.find("no shortfall"), std::string::npos);
  EXPECT_NE(msg.find("does not account for"), std::string::npos);
}

// Both gates pass on paper (physical available and a large-enough contiguous VA
// region) but the allocation still failed -> the driver refused, i.e.
// unaccounted, even without an explicit refusal recorded.
TEST(MemoryManagerOomClassify, BothGatesPassImpliesUnaccounted) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.uncommittedGranules = 50;       // available = 100 MiB
  s.largestRootFreeVA = 100 * kMiB; // VA fits the request
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Unaccounted);
}

TEST(MemoryManagerOomClassify, VaFragmentation) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.uncommittedGranules = 50;      // available = 100 MiB (physical fine)
  s.largestRootFreeVA = 10 * kMiB; // but no contiguous 20 MiB VA region
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::VaFragmentation);
  EXPECT_NE(describeVmmOom(s, c).find("va_factor"), std::string::npos);
}

// A sub-page request still claims whole pages for the physical gate.
TEST(MemoryManagerOomClassify, NeededRoundsUpToWholePages) {
  VmmOomSnapshot s = base(kPage + 1); // 1 byte into a second page
  EXPECT_EQ(s.neededPhysicalBytes(), 2 * kPage);
}

// The VA carve takes the bytes asked for, not the page-rounded physical need.
// A free region that spans the request but not its rounded-up page count is
// sufficient VA, so this is an unaccounted failure, not VA fragmentation.
TEST(MemoryManagerOomClassify, VaGateUsesRequestNotRoundedNeed) {
  VmmOomSnapshot s = base(kPage + 1); // rounds up to 2 pages = 4 MiB
  s.uncommittedGranules = 50;         // available = 100 MiB (physical fine)
  s.largestRootFreeVA = 3 * kMiB;     // >= the request, < the rounded need
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Unaccounted);
}

// Before the arena is reserved there is neither VA nor mapped physical, so the
// budget figures describe nothing. Reporting Genuine here would claim the
// budget is committed when none of it is.
TEST(MemoryManagerOomClassify, NoArenaIsNotGenuine) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.arenaReserved = false;
  // The whole budget is headroom, exactly as computeVmmOomSnapshot reports it
  // when it finds no chunk.
  s.uncommittedGranules = kBudgetGranules;
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::None);
  EXPECT_EQ(c, VmmOomCategory::ArenaUnavailable);
  EXPECT_NE(c, VmmOomCategory::Genuine);
  std::string msg = describeVmmOom(s, c);
  EXPECT_NE(msg.find("not reserved"), std::string::npos);
  EXPECT_NE(msg.find("no allocation can be served"), std::string::npos);
  // The VA lever must not be suggested: the arena, not its factor, is missing.
  EXPECT_EQ(msg.find("va_factor"), std::string::npos);
}

// A refused create is reported even before the arena exists: the reason is
// known outright and does not come from the accounting.
TEST(MemoryManagerOomClassify, RefusedCreateOutranksNoArena) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.arenaReserved = false;
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::CreateRefused),
            VmmOomCategory::Unaccounted);
}

// Mapping is illegal mid-capture, so the same refused map means different
// things depending on whether a capture was in progress.
TEST(MemoryManagerOomClassify, RefusedMapDuringCaptureIsItsOwnCategory) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.uncommittedGranules = 50; // accounting shows plenty
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::MapRefused,
                           /*duringCapture=*/true),
            VmmOomCategory::IllegalDuringCapture);
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::MapRefused,
                           /*duringCapture=*/false),
            VmmOomCategory::Unaccounted);
  std::string msg = describeVmmOom(s, VmmOomCategory::IllegalDuringCapture);
  EXPECT_NE(msg.find("graph capture"), std::string::npos);
  // Must not read as memory pressure: nothing is short here.
  EXPECT_EQ(msg.find("out of memory"), std::string::npos);
}

// A capture in progress does not excuse a refused create: creating physical is
// legal mid-capture, so that remains a genuine shortfall.
TEST(MemoryManagerOomClassify, RefusedCreateDuringCaptureIsStillUnaccounted) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.uncommittedGranules = 50;
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::CreateRefused,
                           /*duringCapture=*/true),
            VmmOomCategory::Unaccounted);
}

// The levers are page-granularity effects, so a world in which they help is
// one where the request rounds up less too. Comparing their sum against the
// page-rounded claim reports a shortfall they cover as an exhaustion.
TEST(MemoryManagerOomClassify, SubPageRequestIsNotGenuineWhenLeversCoverIt) {
  VmmOomSnapshot s = base(101 * kMiB); // not a multiple of the 2 MiB page
  s.inUseGranules = 40;
  s.reclaimableGranules = 30;
  s.pooledGranules = 10;
  s.usedBytes = 59 * kMiB; // trapped = 40*2 - 59 = 21 MiB
  ASSERT_EQ(s.available(), 80 * kMiB);
  ASSERT_EQ(s.trapped(), 21 * kMiB);
  ASSERT_EQ(s.recoverableBySync(), 0u);
  // Levers reach 101 MiB exactly, which covers the request but is one page
  // short of the 102 MiB the request claims as things stand.
  ASSERT_EQ(s.neededPhysicalBytes(), 102 * kMiB);
  EXPECT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Avoidable);
}

// The accounting can show a shortfall and still reach Unaccounted, because the
// manager's own check is local to the region it chose: on a refusal it did
// expect to serve the request. The message must not then deny the shortfall.
TEST(MemoryManagerOomClassify, RefusalBelowAvailableDoesNotDenyTheShortfall) {
  VmmOomSnapshot s = base(100 * kMiB);
  s.inUseGranules = 90;
  s.usedBytes = 180 * kMiB;
  s.reclaimableGranules = 10; // available = 20 MiB, well under the request
  ASSERT_LT(s.available(), s.neededPhysicalBytes());
  VmmOomCategory c = classifyVmmOom(s, VmmBackingFailure::CreateRefused);
  EXPECT_EQ(c, VmmOomCategory::Unaccounted);
  std::string msg = describeVmmOom(s, c);
  EXPECT_EQ(msg.find("shows no shortfall"), std::string::npos);
  EXPECT_NE(msg.find("driver refused"), std::string::npos);
}

TEST(MemoryManagerOomClassify, DumpContainsAllTerms) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.inUseGranules = 40;
  s.pendingGranules = 10;
  s.reclaimableGranules = 5;
  s.pooledGranules = 5;
  s.uncommittedGranules = 40;
  s.usedBytes = 70 * kMiB;
  s.pendingTotalBytes = 18 * kMiB;
  s.largestRootFreeVA = 30 * kMiB;
  std::string d = dumpVmmOom(s);
  for (const char *term : {"vmm_oom",
                           "requested",
                           "page_size",
                           "budget",
                           "used",
                           "available",
                           "recoverable_by_sync",
                           "retained",
                           "retained_bytes",
                           "retained_trapped",
                           "trapped",
                           "pending_total",
                           "largest_root_free_va",
                           "va_size",
                           "free_blocks",
                           "arena_reserved",
                           "page_counts",
                           "in_use",
                           "pending",
                           "reclaimable",
                           "pooled",
                           "uncommitted"})
    EXPECT_NE(d.find(term), std::string::npos) << "missing term: " << term;
}

// Nothing pending: the breakdown is omitted rather than printed empty, so the
// common line stays as short as it was.
TEST(MemoryManagerOomClassify, DumpOmitsOwnerBreakdownWhenNothingIsPending) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.inUseGranules = 100;
  s.usedBytes = s.budget;
  EXPECT_EQ(dumpVmmOom(s).find("pending_by_owner"), std::string::npos);
}

// The breakdown is what turns "50GB is pending" into a name to act on, so the
// largest holder must come first and be tagged when it never releases. The
// list is capped: a long tail of small holders would push the accounting
// fields off the end of a log line, and the tail is never the answer.
TEST(MemoryManagerOomClassify, DumpNamesPendingOwnersLargestFirst) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.inUseGranules = 60;
  s.pendingGranules = 40;
  s.retainedGranules = 40;
  s.usedBytes = 60 * kPage;
  s.pendingTotalBytes = 80 * kMiB;
  s.retainedBytes = 50 * kMiB;
  s.pendingOwners = {{"graphFreeList", 50 * kMiB, 46, /*neverFlushes=*/true},
                     {"freeList", 20 * kMiB, 4, /*neverFlushes=*/false},
                     {"side", 8 * kMiB, 2, /*neverFlushes=*/false},
                     {"hostFreeList", 2 * kMiB, 1, /*neverFlushes=*/false}};
  std::string d = dumpVmmOom(s);
  const size_t graph = d.find("graphFreeList=50MB/46(retained)");
  const size_t ordinary = d.find("freeList=20MB/4");
  const size_t side = d.find("side=8MB/2");
  ASSERT_NE(graph, std::string::npos);
  ASSERT_NE(ordinary, std::string::npos);
  ASSERT_NE(side, std::string::npos);
  EXPECT_LT(graph, ordinary);
  EXPECT_LT(ordinary, side);
  // The fourth holder is rolled up rather than printed.
  EXPECT_EQ(d.find("hostFreeList"), std::string::npos);
  EXPECT_NE(d.find("+1 more"), std::string::npos);
}

// pendingRetryWorthwhile decides whether the allocation path stalls to wait for
// queued frees. It asks what a drain actually hands back, which is a different
// question from which diagnostic category the miss falls in.

// A whole-granule physical shortfall that the queued records cover.
TEST(MemoryManagerPendingRetry, GranulesCloseThePhysicalGap) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB, need 20 MiB
  s.inUseGranules = 98;
  s.usedBytes = 98 * kPage;
  s.largestRootFreeVA = 100 * kMiB; // VA is fine; physical is the problem
  EXPECT_TRUE(withDrainable(s, /*bytes=*/20 * kMiB, /*granules=*/8));
  // The same bytes spread so they free no whole granule buy nothing.
  EXPECT_FALSE(withDrainable(s, /*bytes=*/20 * kMiB, /*granules=*/0));
}

// Trapped bytes satisfy `Avoidable`, but no wait unpins a partially-used page,
// so a shortfall covered only by them must not buy a stall.
TEST(MemoryManagerPendingRetry, TrappedOnlyDoesNotRetry) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.inUseGranules = 98;
  s.usedBytes = 100 * kMiB; // trapped = 96 MiB
  s.largestRootFreeVA = 100 * kMiB;
  ASSERT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Avoidable);
  EXPECT_FALSE(withDrainable(s, /*bytes=*/0, /*granules=*/0));
}

// The VA-shaped miss, which needs a granule larger than base()'s to show: with
// 32 MiB granules a 48 MiB request claims 64 MiB of physical as things stand,
// more than every lever here reaches. It is still servable, because those 48
// MiB can land on VA that is already backed -- a 16 MiB root-free remainder
// trapped in a live granule, beside a 32 MiB pending granule -- once the
// pending block migrates and the two coalesce. `available()` is sufficient but
// not necessary, so the physical shortfall must not veto the VA route.
TEST(MemoryManagerPendingRetry, VaShortfallRetriesDespitePhysicalShortfall) {
  VmmOomSnapshot s;
  s.pageSize = 32 * kMiB;
  s.needed = 48 * kMiB;
  s.arenaReserved = true;
  s.budget = 7 * s.pageSize; // 6 live granules + 1 pending, nothing spare
  s.vaSize = 4 * s.budget;
  s.inUseGranules = 6;
  s.usedBytes = 6 * s.pageSize - 16 * kMiB; // the remainder shares a live
                                            // granule, so it is trapped
  s.pendingGranules = 1;
  s.pendingTotalBytes = 32 * kMiB;
  s.largestRootFreeVA = 16 * kMiB; // < the 48 MiB requested
  ASSERT_EQ(s.available(), 0u);
  ASSERT_EQ(s.neededPhysicalBytes(), 64 * kMiB);
  ASSERT_EQ(s.trapped(), 16 * kMiB);

  // Draining the pending granule yields 32 MiB of physical -- short of the 64
  // MiB the request rounds up to, so the physical route says no -- but it is
  // the VA route that serves this.
  const VmmOomSnapshot pending =
      withDrainableSnapshot(s, /*bytes=*/32 * kMiB, /*granules=*/1);
  ASSERT_LT(pending.available() + pending.drainableGranules * pending.pageSize,
            pending.neededPhysicalBytes());
  EXPECT_TRUE(pendingRetryWorthwhile(pending));

  // Nothing queued to migrate: the root-free region cannot grow, so there is
  // nothing to wait for.
  EXPECT_FALSE(withDrainable(s, /*bytes=*/0, /*granules=*/0));

  // Queued, but far too little to span the gap. Coalescing every migrated byte
  // onto the largest region is the best a drain could do, and it still falls
  // short, so waiting is time spent on an answer that cannot change.
  EXPECT_FALSE(withDrainable(s, /*bytes=*/2 * kMiB, /*granules=*/0));
}

// A graph-capture list queues no gating event, so a drain finds nothing to
// migrate however long it waits. The snapshot already said so through
// `drainable`, which is why the retry never fired in DRIV-690; this pins that
// the new retained accounting does not change the decision.
TEST(MemoryManagerPendingRetry, RetainedMemoryDoesNotBuyAStall) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 2; // available = 4 MiB
  s.pendingGranules = 20;
  s.retainedGranules = 20;
  s.inUseGranules = 78;
  s.usedBytes = 78 * kPage;
  s.pendingTotalBytes = 40 * kMiB;
  s.retainedBytes = 40 * kMiB;
  s.largestRootFreeVA = 100 * kMiB; // VA is fine; physical is the problem
  ASSERT_EQ(classifyVmmOom(s, VmmBackingFailure::None),
            VmmOomCategory::Retained);
  EXPECT_FALSE(withDrainable(s, /*bytes=*/0, /*granules=*/0));
}

// Neither axis short: the physical is held somewhere this manager cannot see,
// and waiting on frees cannot resolve that.
TEST(MemoryManagerPendingRetry, NeitherAxisShortDoesNotRetry) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.reclaimableGranules = 20;       // available = 40 MiB >= need
  s.largestRootFreeVA = 100 * kMiB; // VA fine too
  EXPECT_FALSE(withDrainable(s, /*bytes=*/40 * kMiB, /*granules=*/20));
}

// Without an arena there is neither VA nor physical to reason about, and the
// zeroed snapshot would otherwise read as "everything is short".
TEST(MemoryManagerPendingRetry, NoArenaDoesNotRetry) {
  VmmOomSnapshot s = base(20 * kMiB);
  s.arenaReserved = false;
  EXPECT_FALSE(withDrainable(s, /*bytes=*/40 * kMiB, /*granules=*/20));
}

} // namespace
