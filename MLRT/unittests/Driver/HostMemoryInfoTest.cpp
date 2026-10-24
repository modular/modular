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

#include "MLRT/lib/Driver/DeviceContext/HostMemoryInfo.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <limits>
#include <string>

using namespace M::Driver;

namespace {
constexpr size_t gib = 1024ull * 1024 * 1024;
} // namespace

TEST(HostMemoryInfoTest, IncludesReclaimableMemory) {
  ASSIGN_NO_ERR(Device::MemInfo, info,
                parseHostMemoryInfo("MemTotal: 131072 kB\n"
                                    "MemFree: 4096 kB\n"
                                    "Cached: 98304 kB\n"
                                    "MemAvailable: 110592 kB\n"));
  EXPECT_EQ(info.total, 131072u * 1024);
  EXPECT_EQ(info.free, 110592u * 1024);
}

TEST(HostMemoryInfoTest, FallsBackToFreeMemoryWithoutAvailable) {
  ASSIGN_NO_ERR(Device::MemInfo, info,
                parseHostMemoryInfo("MemTotal: 131072 kB\n"
                                    "MemFree: 4096 kB\n"
                                    "Cached: 98304 kB\n"));
  EXPECT_EQ(info.free, 4096u * 1024);
}

TEST(HostMemoryInfoTest, ZeroAvailabilityDoesNotFallBackToFree) {
  ASSIGN_NO_ERR(Device::MemInfo, info,
                parseHostMemoryInfo("MemTotal: 131072 kB\n"
                                    "MemFree: 4096 kB\n"
                                    "MemAvailable: 0 kB\n"));
  EXPECT_EQ(info.free, 0u);
}

TEST(HostMemoryInfoTest, CapsAvailabilityAtPhysicalMemory) {
  ASSIGN_NO_ERR(Device::MemInfo, info,
                parseHostMemoryInfo("MemTotal: 4096 kB\n"
                                    "MemAvailable: 8192 kB\n"));
  EXPECT_EQ(info.free, info.total);
}

TEST(HostMemoryInfoTest, RejectsMissingCounters) {
  EXPECT_ERR(parseHostMemoryInfo(""));
  EXPECT_ERR(parseHostMemoryInfo("MemAvailable: 4096 kB\n"));
  EXPECT_ERR(parseHostMemoryInfo("MemTotal: 4096 kB\n"));
  EXPECT_ERR(parseHostMemoryInfo("MemTotal: 0 kB\nMemAvailable: 0 kB\n"));
}

TEST(HostMemoryInfoTest, RejectsMalformedCounters) {
  EXPECT_ERR(parseHostMemoryInfo("MemTotal: -1 kB\nMemAvailable: 4096 kB\n"));
  EXPECT_ERR(parseHostMemoryInfo("MemTotal: 8192 kB\nMemAvailable: nope kB\n"));
  EXPECT_ERR(parseHostMemoryInfo("MemTotal: 8192 MB\nMemAvailable: 4096 kB\n"));
  EXPECT_ERR(parseHostMemoryInfo("MemTotal: 8192 kB\nMemAvailable: 4096\n"));
}

TEST(HostMemoryInfoTest, RejectsByteSizeOverflow) {
  std::string kb =
      std::to_string(std::numeric_limits<size_t>::max() / 1024 + 1);
  EXPECT_ERR(
      parseHostMemoryInfo("MemTotal: " + kb + " kB\nMemAvailable: 4096 kB\n"));
}

TEST(HostMemoryInfoTest, CgroupLimitClampsHeadroom) {
  // A 32 GiB limit with 4 GiB charged, of which 2 GiB is reclaimable cache
  // (3 GiB file minus 1 GiB shmem): 30 GiB may still be charged.
  auto headroom = parseCgroupHeadroom("34359738368", "4294967296",
                                      "file 3221225472\nshmem 1073741824\n",
                                      "file", "shmem", true);
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, size_t(30) * gib);
}

TEST(HostMemoryInfoTest, CgroupV1ClampsWithLegacyStatKeys) {
  auto headroom =
      parseCgroupHeadroom("34359738368", "4294967296",
                          "total_cache 3221225472\ntotal_shmem 1073741824\n",
                          "total_cache", "total_shmem", false);
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, size_t(30) * gib);
}

TEST(HostMemoryInfoTest, CgroupV2MaxIsUnlimited) {
  EXPECT_FALSE(parseCgroupHeadroom("max", "1024", "", "file", "shmem", true)
                   .has_value());
}

TEST(HostMemoryInfoTest, CgroupV1SentinelIsUnlimited) {
  EXPECT_FALSE(parseCgroupHeadroom("9223372036854771712", "1024", "",
                                   "total_cache", "total_shmem", false)
                   .has_value());
}

TEST(HostMemoryInfoTest, CgroupUsageOverLimitReportsZero) {
  // Usage can briefly exceed the limit during reclaim.
  auto headroom = parseCgroupHeadroom("1024", "4096", "file 0\nshmem 0\n",
                                      "file", "shmem", true);
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, 0u);
}

TEST(HostMemoryInfoTest, SharedPoolSubstitutesHostAvailability) {
  Device::MemInfo deviceInfo, hostInfo;
  deviceInfo.total = 100 * gib;
  deviceInfo.free = 30 * gib;
  hostInfo.total = 100 * gib;
  hostInfo.free = 70 * gib;
  ASSIGN_NO_ERR(Device::MemInfo, info,
                getHostUnifiedMemoryInfo(deviceInfo, hostInfo));
  EXPECT_EQ(info.total, 100 * gib);
  EXPECT_EQ(info.free, 70 * gib);
}

TEST(HostMemoryInfoTest, CarveoutStillCountsAsShared) {
  Device::MemInfo deviceInfo, hostInfo;
  deviceInfo.total = 85 * gib;
  deviceInfo.free = 10 * gib;
  hostInfo.total = 100 * gib;
  hostInfo.free = 60 * gib;
  ASSIGN_NO_ERR(Device::MemInfo, info,
                getHostUnifiedMemoryInfo(deviceInfo, hostInfo));
  EXPECT_EQ(info.free, 60 * gib);
}

TEST(HostMemoryInfoTest, PrivateMemoryKeepsDriverFigures) {
  // Integrated parts with their own memory, like MI300A, keep the driver's
  // figures.
  Device::MemInfo deviceInfo, hostInfo;
  deviceInfo.total = 128 * gib;
  deviceInfo.free = 100 * gib;
  hostInfo.total = 900 * gib;
  hostInfo.free = 700 * gib;
  ASSIGN_NO_ERR(Device::MemInfo, info,
                getHostUnifiedMemoryInfo(deviceInfo, hostInfo));
  EXPECT_EQ(info.total, 128 * gib);
  EXPECT_EQ(info.free, 100 * gib);
}

TEST(HostMemoryInfoTest, BelowSharedThresholdKeepsDriverFigures) {
  Device::MemInfo deviceInfo, hostInfo;
  deviceInfo.total = 79 * gib;
  deviceInfo.free = 10 * gib;
  hostInfo.total = 100 * gib;
  hostInfo.free = 60 * gib;
  ASSIGN_NO_ERR(Device::MemInfo, info,
                getHostUnifiedMemoryInfo(deviceInfo, hostInfo));
  EXPECT_EQ(info.free, 10 * gib);
}

TEST(HostMemoryInfoTest, RespectsDriverCapacity) {
  Device::MemInfo deviceInfo, hostInfo;
  deviceInfo.total = 1024;
  deviceInfo.free = 0;
  hostInfo.total = 1024;
  hostInfo.free = 2048;
  ASSIGN_NO_ERR(Device::MemInfo, info,
                getHostUnifiedMemoryInfo(deviceInfo, hostInfo));
  EXPECT_EQ(info.total, 1024u);
  EXPECT_EQ(info.free, 1024u);
}
