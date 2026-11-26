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
// The memory manager over fake backing and fake streams (MemoryManagerFakes.h).
// These tests cover what a GPU test cannot: the driver calls the manager makes,
// and its rollback when the driver refuses a reservation, create, map or access
// grant. The GPU suites cover the paths that need no injected refusal.
//
// Each test asserts something only its path produces, so a test that stopped
// reaching its path would fail rather than pass vacuously. After every test the
// fixture clears the manager and checks that the fakes hold nothing, so a leak
// fails the test that caused it.
//
//===----------------------------------------------------------------------===//

#include "MemoryManagerFakes.h"

#include "MLRT/lib/Driver/MemoryManager/MemoryManager.h"
#include "TestUtils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

namespace {

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;
// The VMM page, which is also the fake driver's granularity.
constexpr size_t kPage = 2 * kMiB;
constexpr AllocatorContext::AllocMode kSync = AllocatorContext::AllocMode::SYNC;

// The error of an allocation that must miss.
std::string missError(ErrorOr<void *> result) {
  if (!result.isError()) {
    ADD_FAILURE() << "expected the allocation to miss, but it returned "
                  << result.takeValue();
    return "";
  }
  return result.getError();
}

class MemoryManagerFakeBackingTest : public ::testing::Test {
protected:
  struct Child {
    std::shared_ptr<FakeStream> stream;
    std::unique_ptr<MemoryManagerFreeList> list;
  };

  void TearDown() override {
    if (vmm_ != nullptr)
      EXPECT_EQ(vmm_->mappingsWithoutAccess(), 0u)
          << "the manager handed out a mapping it never made accessible";
    // The order Device::clearMemoryManager tears down in: the per-stream free
    // lists first, then the manager is cleared, then destroyed.
    for (std::unique_ptr<Child> &child : children_)
      child->list.reset();
    if (mm_ != nullptr) {
      if (uint64_t live = mm_->getAllocationStats().liveBytes; live != 0) {
        // clear() aborts on a live allocation, which would hide this failure
        // and every test after it.
        ADD_FAILURE() << "the test left " << live << " bytes allocated";
        (void)mm_.release();
        return;
      }
      mm_->clear();
      if (device_ != nullptr)
        device_->expectNothingLive();
      if (vmm_ != nullptr)
        vmm_->expectNothingLive();
      mm_.reset();
    }
  }

  static MemoryManagerOptions commonOptions() {
    MemoryManagerOptions options;
    options.alignment = 256;
    // A miss fails rather than falling through to the driver, and does not
    // wait out pending frees.
    options.useOnlyMemoryManagerMode = true;
    options.drainRetries = 0;
    options.drainBudgetMs = 1000;
    options.selfCheck = true;
    options.label = "fake[0]";
    return options;
  }

  // A single-arena VMM manager with a budget of `budgetPages` pages.
  static MemoryManagerOptions vmmOptions(size_t budgetPages) {
    MemoryManagerOptions options = commonOptions();
    options.maxSize = budgetPages * kPage;
    options.chunkSize = options.maxSize;
    options.vmmEnabled = true;
    options.vmmPageSize = kPage;
    options.vmmVaFactor = 4;
    return options;
  }

  static MemoryManagerOptions plainOptions(size_t maxSize, size_t chunkSize) {
    MemoryManagerOptions options = commonOptions();
    options.maxSize = maxSize;
    options.chunkSize = chunkSize;
    return options;
  }

  void createManager(MemoryManagerOptions options) {
    AllocatorBacking backing;
    if (options.vmmEnabled) {
      auto vmm = std::make_unique<FakeAllocatorVmm>(options.vmmPageSize);
      vmm->capturingWhen([this] { return anyStreamCapturing(); });
      vmm_ = vmm.get();
      backing = std::move(vmm);
    } else {
      auto device = std::make_unique<FakeAllocatorChunks>();
      device_ = device.get();
      backing = std::move(device);
    }
    mm_ = MemoryManager::create(std::move(options), std::move(backing));
  }

  bool anyStreamCapturing() const {
    for (const std::shared_ptr<FakeStream> &stream : streams_)
      if (stream->capturing)
        return true;
    return false;
  }

  // A free list on its own fake stream under the root free list, as a
  // context's free list is.
  Child &addChild(std::string label) {
    auto child = std::make_unique<Child>();
    child->stream = std::make_shared<FakeStream>();
    streams_.push_back(child->stream);
    child->list = std::make_unique<MemoryManagerFreeList>(
        std::make_unique<FakeAllocatorContext>(child->stream), *mm_,
        mm_->freeList(), /*disableFlush=*/false, std::move(label),
        /*redzones=*/false, /*poison=*/false);
    // A context's poll drains its own free list's completed frees.
    child->stream->onPoll = [this, list = child->list.get()] {
      return mm_->drainCompletedFrees(list);
    };
    children_.push_back(std::move(child));
    return *children_.back();
  }

  std::unique_ptr<MemoryManager> mm_;
  // Owned by `mm_`.
  FakeAllocatorChunks *device_ = nullptr;
  FakeAllocatorVmm *vmm_ = nullptr;
  std::vector<std::shared_ptr<FakeStream>> streams_;
  std::vector<std::unique_ptr<Child>> children_;
};

// useMaxSize_ is process-global, and a plain manager's first allocation past
// its chunk size sets it, after which every plain manager in the process grows
// its chunks to the budget. No test here makes such an allocation, so the
// sizes below hold in any order. MemoryManagerChunkGrowthTest covers that
// heuristic in a process of its own.
TEST_F(MemoryManagerFakeBackingTest, PlainChunksGrowWithinBudgetAndDeviceFree) {
  createManager(plainOptions(/*maxSize=*/kMiB, /*chunkSize=*/256 * kKiB));
  MemoryManagerFreeList &root = *mm_->freeList();

  // The first block fills the first chunk, so the second needs another.
  ASSIGN_NO_ERR(void *, a, root.allocate(256 * kKiB, kSync));
  ASSIGN_NO_ERR(void *, b, root.allocate(128 * kKiB, kSync));
  EXPECT_EQ(device_->chunkAllocSizes(),
            (std::vector<size_t>{256 * kKiB, 256 * kKiB}));

  // Past the chunk size, the chunk grows to the request.
  ASSIGN_NO_ERR(void *, c, root.allocate(384 * kKiB, kSync));
  ASSIGN_NO_ERR(void *, d, root.allocate(128 * kKiB, kSync));
  EXPECT_EQ(device_->chunkAllocSizes(),
            (std::vector<size_t>{256 * kKiB, 256 * kKiB, 384 * kKiB}));

  device_->refuseChunkAllocs(true);
  std::string refused = missError(root.allocate(64 * kKiB, kSync));
  EXPECT_THAT(refused, HasSubstr("last_chunk_alloc_failure: chunkAlloc("));
  EXPECT_THAT(refused, HasSubstr("failed with fake: chunkAlloc refused"));
  EXPECT_EQ(device_->refusedChunkAllocs(), 1u);
  device_->refuseChunkAllocs(false);

  // The next chunk is capped at what the device has free.
  device_->setMemoryInfo(/*total=*/64 * kMiB, /*free=*/96 * kKiB);
  ASSIGN_NO_ERR(void *, e, root.allocate(64 * kKiB, kSync));
  EXPECT_EQ(device_->chunkAllocSizes().back(), 96 * kKiB);

  // The 32 KiB left in the budget is below the smallest chunk worth taking, so
  // the driver is not asked at all.
  std::string exhausted = missError(root.allocate(64 * kKiB, kSync));
  EXPECT_THAT(exhausted,
              HasSubstr("remaining memory manager capacity is insufficient"));
  EXPECT_EQ(device_->chunkAllocSizes().size(), 4u);
  EXPECT_EQ(device_->refusedChunkAllocs(), 1u);

  for (void *ptr : {a, b, c, d, e})
    EXPECT_NO_ERR(root.free(ptr));
}

// The arena reserves address space on first use, several times the budget,
// and creates physical pages only as allocations reach them. A freed page stays
// mapped until the budget runs out and it is reclaimed.
TEST_F(MemoryManagerFakeBackingTest, ArenaIsReservedLazilyAndBackedPerPage) {
  createManager(vmmOptions(/*budgetPages=*/4));
  MemoryManagerFreeList &root = *mm_->freeList();
  EXPECT_TRUE(vmm_->reserveRequests().empty());

  ASSIGN_NO_ERR(void *, a, root.allocate(kPage, kSync));
  EXPECT_EQ(vmm_->reserveRequests(), std::vector<size_t>{16 * kPage});
  EXPECT_EQ(vmm_->createCount(), 1u);
  EXPECT_NE(vmm_->handleAt(a), 0u);

  // Two half-page blocks share the second page.
  ASSIGN_NO_ERR(void *, b, root.allocate(kPage / 2, kSync));
  ASSIGN_NO_ERR(void *, c, root.allocate(kPage / 2, kSync));
  EXPECT_EQ(vmm_->createCount(), 2u);
  EXPECT_EQ(mm_->createdPhysicalBytesForTesting(), 2 * kPage);

  for (void *ptr : {a, b, c})
    EXPECT_NO_ERR(root.free(ptr));
  EXPECT_EQ(vmm_->liveMappings(), 2u);
  EXPECT_EQ(vmm_->unmapCount(), 0u);
}

// With the whole budget created, a request at unbacked address space takes its
// pages from root-free ones: unmapped there, mapped here, none created.
TEST_F(MemoryManagerFakeBackingTest, FullBudgetIsReclaimedFromRootFreePages) {
  createManager(vmmOptions(/*budgetPages=*/4));
  MemoryManagerFreeList &root = *mm_->freeList();

  std::vector<void *> pages;
  for (int i = 0; i < 4; ++i) {
    ASSIGN_NO_ERR(void *, page, root.allocate(kPage, kSync));
    pages.push_back(page);
  }
  char *base = static_cast<char *>(pages[0]);
  const uint64_t freed0 = vmm_->handleAt(pages[0]);
  const uint64_t freed2 = vmm_->handleAt(pages[2]);
  // Two separate free pages, so neither fits a two-page request.
  EXPECT_NO_ERR(root.free(pages[0]));
  EXPECT_NO_ERR(root.free(pages[2]));

  ASSIGN_NO_ERR(void *, big, root.allocate(2 * kPage, kSync));
  EXPECT_EQ(big, base + 4 * kPage);
  EXPECT_EQ(vmm_->createCount(), 4u);
  EXPECT_EQ(vmm_->handleAt(base), 0u);
  EXPECT_EQ(vmm_->handleAt(base + 2 * kPage), 0u);
  EXPECT_THAT((std::vector<uint64_t>{vmm_->handleAt(base + 4 * kPage),
                                     vmm_->handleAt(base + 5 * kPage)}),
              UnorderedElementsAre(freed0, freed2));
  EXPECT_EQ(mm_->createdPhysicalBytesForTesting(), 4 * kPage);

  for (void *ptr : {big, pages[1], pages[3]})
    EXPECT_NO_ERR(root.free(ptr));
}

// A free landing while the list's previous flush is in flight joins the list
// being filled, which no event covers. deferCachedToParent queues it.
TEST_F(MemoryManagerFakeBackingTest, FreeDuringAFlushWaitsForDeferCached) {
  createManager(plainOptions(kMiB, kMiB));
  MemoryManagerFreeList &root = *mm_->freeList();
  Child &child = addChild("child");

  ASSIGN_NO_ERR(void *, first, child.list->allocate(128 * kKiB, kSync));
  // Keeps the two frees from coalescing into one block.
  ASSIGN_NO_ERR(void *, spacer, root.allocate(128 * kKiB, kSync));
  ASSIGN_NO_ERR(void *, second, child.list->allocate(128 * kKiB, kSync));
  EXPECT_NO_ERR(child.list->free(first));
  EXPECT_NO_ERR(child.list->free(second));
  EXPECT_EQ(child.stream->eventsRecorded(), 1u);

  child.stream->runAll();
  EXPECT_NO_ERR(mm_->drainCompletedFrees());
  EXPECT_FALSE(child.list->empty()) << "the second free has no event";

  EXPECT_TRUE(child.list->deferCachedToParent());
  EXPECT_EQ(child.stream->eventsRecorded(), 2u);
  EXPECT_NO_ERR(mm_->drainCompletedFrees());
  EXPECT_FALSE(child.list->empty());
  child.stream->runAll();
  EXPECT_NO_ERR(mm_->drainCompletedFrees());
  EXPECT_TRUE(child.list->empty());

  EXPECT_NO_ERR(root.free(spacer));
}

TEST_F(MemoryManagerFakeBackingTest, RefusedRemapMapRestoresTheStreamsPages) {
  createManager(vmmOptions(/*budgetPages=*/4));
  Child &child = addChild("remap");

  ASSIGN_NO_ERR(void *, small, child.list->allocate(2 * kPage, kSync));
  char *base = static_cast<char *>(small);
  const uint64_t h0 = vmm_->handleAt(base);
  const uint64_t h1 = vmm_->handleAt(base + kPage);
  EXPECT_NO_ERR(child.list->free(small));

  // The first reused page maps into the new range, the second is refused.
  vmm_->refuseMapsAfter(1);
  const size_t unmapsBefore = vmm_->unmapCount();
  std::string error = missError(child.list->allocate(3 * kPage, kSync));
  EXPECT_THAT(error, HasSubstr("(unaccounted)"));
  EXPECT_EQ(vmm_->refusedMaps(), 1u);
  EXPECT_EQ(vmm_->unmapCount(), unmapsBefore + 1);
  EXPECT_EQ(vmm_->handleAt(base + 2 * kPage), 0u);
  EXPECT_EQ(vmm_->aliasedHandles(), 0u);
  EXPECT_EQ(vmm_->handleAt(base), h0);
  EXPECT_EQ(vmm_->handleAt(base + kPage), h1);
  EXPECT_EQ(mm_->streamRemaps(), 0u);
  EXPECT_EQ(mm_->retiringRangesForTesting(), 0u);
  EXPECT_EQ(mm_->createdPhysicalBytesForTesting(), 2 * kPage);
  EXPECT_FALSE(child.list->empty());
  vmm_->refuseMapsAfter(-1);
}

TEST_F(MemoryManagerFakeBackingTest,
       RefusedRemapAccessRestoresTheStreamsPages) {
  createManager(vmmOptions(/*budgetPages=*/4));
  Child &child = addChild("remap");

  ASSIGN_NO_ERR(void *, small, child.list->allocate(2 * kPage, kSync));
  char *base = static_cast<char *>(small);
  const uint64_t h0 = vmm_->handleAt(base);
  const uint64_t h1 = vmm_->handleAt(base + kPage);
  EXPECT_NO_ERR(child.list->free(small));

  // Both reused pages map into the new range, then their access is refused.
  vmm_->refuseAccessAfter(0);
  const size_t unmapsBefore = vmm_->unmapCount();
  std::string error = missError(child.list->allocate(3 * kPage, kSync));
  EXPECT_THAT(error, HasSubstr("(unaccounted)"));
  EXPECT_EQ(vmm_->refusedAccess(), 1u);
  EXPECT_EQ(vmm_->unmapCount(), unmapsBefore + 2);
  EXPECT_EQ(vmm_->handleAt(base + 2 * kPage), 0u);
  EXPECT_EQ(vmm_->aliasedHandles(), 0u);
  EXPECT_EQ(vmm_->handleAt(base), h0);
  EXPECT_EQ(vmm_->handleAt(base + kPage), h1);
  EXPECT_EQ(mm_->streamRemaps(), 0u);
  EXPECT_EQ(mm_->retiringRangesForTesting(), 0u);
  EXPECT_FALSE(child.list->empty());
  vmm_->refuseAccessAfter(-1);
}

TEST_F(MemoryManagerFakeBackingTest, RefusedReservationIsArenaUnavailable) {
  createManager(vmmOptions(/*budgetPages=*/4));
  vmm_->refuseReserves();

  std::string error = missError(mm_->freeList()->allocate(kPage, kSync));
  EXPECT_THAT(error, HasSubstr("(arena unavailable)"));
  EXPECT_THAT(error, HasSubstr("vmmAddressReserve("));
  // The budget-sized fallback was tried too.
  EXPECT_EQ(vmm_->reserveRequests(),
            (std::vector<size_t>{16 * kPage, 4 * kPage}));
  EXPECT_FALSE(mm_->oomSnapshotForTesting(kPage).arenaReserved);
}

// Mapping is illegal mid-capture, so a map the driver refuses then is a gap in
// the pre-capture backing, not a shortage. Backing the budget before the
// capture closes it.
TEST_F(MemoryManagerFakeBackingTest, RefusedMapDuringCaptureIsIllegal) {
  createManager(vmmOptions(/*budgetPages=*/4));
  Child &child = addChild("capture");

  child.stream->capturing = true;
  std::string error = missError(child.list->allocate(kPage, kSync));
  EXPECT_THAT(error, HasSubstr("(illegal during graph capture)"));
  EXPECT_EQ(vmm_->refusedMaps(), 1u);
  EXPECT_EQ(vmm_->liveMappings(), 0u);

  child.stream->capturing = false;
  EXPECT_NO_ERR(mm_->preBackForCapture());
  EXPECT_EQ(vmm_->liveMappings(), 4u);

  child.stream->capturing = true;
  ASSIGN_NO_ERR(void *, block, child.list->allocate(kPage, kSync));
  EXPECT_EQ(vmm_->refusedMaps(), 1u) << "the capture needed no map";
  EXPECT_NO_ERR(child.list->free(block));
  child.stream->capturing = false;
}

// The budget had room, so a refused create means the physical is held outside
// the manager. Every page mapped for the request is unmapped again, and its
// handle kept for the next request.
TEST_F(MemoryManagerFakeBackingTest, RefusedCreateRollsBackAndIsUnaccounted) {
  createManager(vmmOptions(/*budgetPages=*/4));
  MemoryManagerFreeList &root = *mm_->freeList();

  vmm_->refuseCreatesAfter(1);
  std::string error = missError(root.allocate(3 * kPage, kSync));
  EXPECT_THAT(error, HasSubstr("(unaccounted)"));
  EXPECT_EQ(vmm_->refusedCreates(), 1u);
  EXPECT_EQ(vmm_->liveMappings(), 0u);
  EXPECT_EQ(vmm_->liveHandles(), 1u);
  EXPECT_EQ(mm_->createdPhysicalBytesForTesting(), kPage);

  vmm_->refuseCreatesAfter(-1);
  ASSIGN_NO_ERR(void *, block, root.allocate(3 * kPage, kSync));
  EXPECT_EQ(vmm_->createCount(), 3u) << "the kept handle backs one page";
  EXPECT_EQ(vmm_->liveMappings(), 3u);
  EXPECT_NO_ERR(root.free(block));
}

TEST_F(MemoryManagerFakeBackingTest, RefusedMapRollsBackAndIsUnaccounted) {
  createManager(vmmOptions(/*budgetPages=*/4));
  MemoryManagerFreeList &root = *mm_->freeList();

  vmm_->refuseMapsAfter(1);
  std::string error = missError(root.allocate(3 * kPage, kSync));
  EXPECT_THAT(error, HasSubstr("(unaccounted)"));
  EXPECT_EQ(vmm_->refusedMaps(), 1u);
  EXPECT_EQ(vmm_->liveMappings(), 0u);
  EXPECT_EQ(vmm_->liveHandles(), 2u);
  EXPECT_EQ(mm_->createdPhysicalBytesForTesting(), 2 * kPage);

  vmm_->refuseMapsAfter(-1);
  ASSIGN_NO_ERR(void *, block, root.allocate(3 * kPage, kSync));
  EXPECT_EQ(vmm_->createCount(), 3u) << "both kept handles back a page";
  EXPECT_EQ(vmm_->liveMappings(), 3u);
  EXPECT_NO_ERR(root.free(block));
}

} // namespace
