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

using namespace M::Driver;

namespace {
constexpr size_t gib = 1024ull * 1024 * 1024;
} // namespace

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
