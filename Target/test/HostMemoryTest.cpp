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

#include "Target/lib/Support/HostMemory.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <gtest/gtest.h>
#include <limits>
#include <string>

#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

using namespace M::Target;

namespace {
constexpr size_t gib = 1024ull * 1024 * 1024;

HostMemory parse(llvm::StringRef meminfo) {
  auto memory = parseLinuxMemInfo(meminfo);
  if (!memory) {
    ADD_FAILURE() << llvm::toString(memory.takeError());
    return {};
  }
  return *memory;
}

bool rejects(llvm::StringRef meminfo) {
  auto memory = parseLinuxMemInfo(meminfo);
  if (memory)
    return false;
  llvm::consumeError(memory.takeError());
  return true;
}

size_t physicalMemory() {
#ifdef __APPLE__
  uint64_t total = 0;
  size_t length = sizeof(total);
  sysctlbyname("hw.memsize", &total, &length, nullptr, 0);
  return total;
#else
  return size_t(sysconf(_SC_PHYS_PAGES)) * size_t(sysconf(_SC_PAGE_SIZE));
#endif
}
} // namespace

TEST(HostMemoryTest, IncludesReclaimableMemory) {
  HostMemory info = parse("MemTotal: 131072 kB\n"
                          "MemFree: 4096 kB\n"
                          "Cached: 98304 kB\n"
                          "MemAvailable: 110592 kB\n");
  EXPECT_EQ(info.total, 131072u * 1024);
  EXPECT_EQ(info.available, 110592u * 1024);
}

TEST(HostMemoryTest, FallsBackToFreeMemoryWithoutAvailable) {
  HostMemory info = parse("MemTotal: 131072 kB\n"
                          "MemFree: 4096 kB\n"
                          "Cached: 98304 kB\n");
  EXPECT_EQ(info.available, 4096u * 1024);
}

TEST(HostMemoryTest, ZeroAvailabilityDoesNotFallBackToFree) {
  HostMemory info = parse("MemTotal: 131072 kB\n"
                          "MemFree: 4096 kB\n"
                          "MemAvailable: 0 kB\n");
  EXPECT_EQ(info.available, 0u);
}

TEST(HostMemoryTest, CapsAvailabilityAtPhysicalMemory) {
  HostMemory info = parse("MemTotal: 4096 kB\n"
                          "MemAvailable: 8192 kB\n");
  EXPECT_EQ(info.available, info.total);
}

TEST(HostMemoryTest, RejectsMissingCounters) {
  EXPECT_TRUE(rejects(""));
  EXPECT_TRUE(rejects("MemAvailable: 4096 kB\n"));
  EXPECT_TRUE(rejects("MemTotal: 4096 kB\n"));
  EXPECT_TRUE(rejects("MemTotal: 0 kB\nMemAvailable: 0 kB\n"));
}

TEST(HostMemoryTest, RejectsMalformedCounters) {
  EXPECT_TRUE(rejects("MemTotal: -1 kB\nMemAvailable: 4096 kB\n"));
  EXPECT_TRUE(rejects("MemTotal: 8192 kB\nMemAvailable: nope kB\n"));
  EXPECT_TRUE(rejects("MemTotal: 8192 MB\nMemAvailable: 4096 kB\n"));
  EXPECT_TRUE(rejects("MemTotal: 8192 kB\nMemAvailable: 4096\n"));
}

TEST(HostMemoryTest, RejectsByteSizeOverflow) {
  std::string kb =
      std::to_string(std::numeric_limits<size_t>::max() / 1024 + 1);
  EXPECT_TRUE(rejects("MemTotal: " + kb + " kB\nMemAvailable: 4096 kB\n"));
}

TEST(HostMemoryTest, CgroupLimitClampsHeadroom) {
  // A 32 GiB limit with 4 GiB charged, of which 2 GiB is reclaimable cache
  // (3 GiB file minus 1 GiB shmem): 30 GiB may still be charged.
  auto headroom = parseCgroupHeadroom("34359738368", "4294967296",
                                      "file 3221225472\nshmem 1073741824\n",
                                      "file", "shmem", true);
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, size_t(30) * gib);
}

TEST(HostMemoryTest, CgroupV1ClampsWithLegacyStatKeys) {
  auto headroom =
      parseCgroupHeadroom("34359738368", "4294967296",
                          "total_cache 3221225472\ntotal_shmem 1073741824\n",
                          "total_cache", "total_shmem", false);
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, size_t(30) * gib);
}

TEST(HostMemoryTest, CgroupV2MaxIsUnlimited) {
  EXPECT_FALSE(parseCgroupHeadroom("max", "1024", "", "file", "shmem", true)
                   .has_value());
}

TEST(HostMemoryTest, CgroupV1SentinelIsUnlimited) {
  EXPECT_FALSE(parseCgroupHeadroom("9223372036854771712", "1024", "",
                                   "total_cache", "total_shmem", false)
                   .has_value());
}

TEST(HostMemoryTest, CgroupUsageOverLimitReportsZero) {
  // Usage can briefly exceed the limit during reclaim.
  auto headroom = parseCgroupHeadroom("1024", "4096", "file 0\nshmem 0\n",
                                      "file", "shmem", true);
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, 0u);
}

TEST(HostMemoryTest, DarwinCountsFreeAndInactivePages) {
  HostMemory memory =
      darwinHostMemory(/*total=*/48 * gib, /*pageSize=*/16384,
                       /*freePages=*/32768, /*inactivePages=*/65536);
  EXPECT_EQ(memory.total, 48 * gib);
  EXPECT_EQ(memory.available, (32768u + 65536u) * 16384u);
}

TEST(HostMemoryTest, DarwinCapsAvailabilityAtPhysicalMemory) {
  HostMemory memory = darwinHostMemory(/*total=*/gib, /*pageSize=*/16384,
                                       /*freePages=*/65536,
                                       /*inactivePages=*/65536);
  EXPECT_EQ(memory.available, gib);
}

// Container runtimes such as LXCFS rewrite MemTotal to the container limit,
// and a cgroup at its limit has no headroom, so only bounds are checked.
TEST(HostMemoryTest, QueryReportsSystemWideAvailability) {
  auto memory = queryHostMemory();
  ASSERT_TRUE(bool(memory)) << llvm::toString(memory.takeError());
  EXPECT_GT(memory->total, 0u);
  EXPECT_LE(memory->total, physicalMemory());
  EXPECT_LE(memory->available, memory->total);
}

TEST(HostMemoryTest, ReadsKernelFilesThatReportZeroSize) {
#ifndef __linux__
  GTEST_SKIP() << "procfs is Linux-only";
#else
  auto cgroup = readKernelFile("/proc/self/cgroup");
  ASSERT_TRUE(cgroup.has_value());
  EXPECT_FALSE(cgroup->empty());
  auto meminfo = readKernelFile("/proc/meminfo");
  ASSERT_TRUE(meminfo.has_value());
  EXPECT_GT(parse(*meminfo).total, 0u);
#endif
}
