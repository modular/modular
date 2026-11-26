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
// GPU-free tests for the VMM granule the arena uses when
// `memory_manager_vmm_page_mb` is unset. The default used to be a fixed 256
// MiB, which on a 109.69 GiB arena is 439 granules: a request the free list
// cannot assemble from whole granules fails while the ledger reports the memory
// free, and the device refuses it. These pin the derived granule and the
// bounds.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/MemoryManager/MemoryManager.h"

#include "gtest/gtest.h"

#include <cstddef>

using namespace M::Driver;

namespace {

constexpr size_t kMiB = size_t(1024) * 1024;
constexpr size_t kGiB = size_t(1024) * kMiB;

// What a B200 reports as its device-free budget at the utilization the serving
// configuration asked for: 109.69 GiB.
constexpr size_t kB200Arena = size_t(117'776'000'000);

// What a device maps in one piece; NVIDIA and HIP both report 2 MiB.
constexpr size_t kGranularity = size_t(2) * kMiB;

} // namespace

TEST(VmmPageSize, TheB200ArenaGets28MiB) {
  const size_t page = deriveVmmPageSize(kB200Arena, kGranularity);
  EXPECT_EQ(page, size_t(28) * kMiB);
  // The granule that motivated the change: 439 granules is too coarse for the
  // free list to place a large request out of.
  EXPECT_LT(kB200Arena / (size_t(256) * kMiB), 512u);
  EXPECT_GE(kB200Arena / page, 4000u);
}

TEST(VmmPageSize, FloorsAtTwoMiB) {
  EXPECT_EQ(deriveVmmPageSize(size_t(1) * kGiB, kGranularity),
            size_t(2) * kMiB);
  EXPECT_EQ(deriveVmmPageSize(size_t(4) * kGiB, kGranularity),
            size_t(2) * kMiB);
}

TEST(VmmPageSize, CeilingsAt64MiB) {
  // The HIP ceiling case (eight MI355X) is a 1.6 TiB arena summed over
  // devices; one device's share is still large enough to hit the ceiling.
  EXPECT_EQ(deriveVmmPageSize(size_t(400) * kGiB, kGranularity),
            size_t(64) * kMiB);
}

// Between the floor and the ceiling the target holds: the arena stays a few
// thousand granules wide however large it is.
TEST(VmmPageSize, TracksTheArenaBetweenTheBounds) {
  for (const size_t budgetGiB :
       {size_t(8), size_t(16), size_t(24), size_t(48), size_t(96), size_t(109),
        size_t(180), size_t(256)}) {
    const size_t budget = budgetGiB * kGiB;
    const size_t page = deriveVmmPageSize(budget, kGranularity);
    EXPECT_EQ(page % kGranularity, 0u) << budgetGiB << " GiB";
    EXPECT_GE(budget / page, 2048u) << budgetGiB << " GiB";
    EXPECT_LE(budget / page, 4096u) << budgetGiB << " GiB";
  }
}

// The derived granule is never below the device's own granularity.
TEST(VmmPageSize, RoundsUpToTheDeviceGranularity) {
  const size_t coarse = size_t(32) * kMiB;
  EXPECT_EQ(deriveVmmPageSize(size_t(8) * kGiB, coarse), coarse);
  EXPECT_EQ(deriveVmmPageSize(size_t(48) * kGiB, coarse), coarse);
}

// A backend that reports no granularity still gets a usable granule.
TEST(VmmPageSize, NoGranularityFloors) {
  EXPECT_EQ(deriveVmmPageSize(size_t(96) * kGiB, 0), size_t(2) * kMiB);
}
