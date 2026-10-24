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

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/HostMemoryInfo.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <iostream>

using namespace M::Driver;

TEST(HostUnifiedMemoryTest, ReportsReclaimableHostMemoryAndAllocatorBudget) {
#if !defined(__linux__)
  GTEST_SKIP() << "Linux host memory accounting is not used on this platform";
#else
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(bool, unified, ctx->device().isHostUnified());
  if (!unified)
    GTEST_SKIP() << "Device does not share physical memory with the host";

  ASSIGN_NO_ERR(Device::MemInfo, deviceInfo, ctx->device().getMemoryInfo());
  ASSIGN_NO_ERR(Device::MemInfo, hostInfo,
                getHostUnifiedMemoryInfo(deviceInfo));
  ASSIGN_NO_ERR(Device::MemInfo, allocatorInfo, ctx->getMemoryInfo());

  // deviceInfo and hostInfo are consecutive /proc/meminfo reads, so only
  // other processes' activity separates them. The allocator budget is fixed
  // at device creation, so availability may have grown since; it gets the
  // wider bound.
  constexpr size_t drift = 1024ull * 1024 * 1024;
  constexpr size_t budgetDrift = 2ull * 1024 * 1024 * 1024;
  EXPECT_LE(deviceInfo.free, deviceInfo.total);
  EXPECT_GE(deviceInfo.free + drift, hostInfo.free);
  EXPECT_GE(hostInfo.free + drift, deviceInfo.free);
  EXPECT_GE(allocatorInfo.total + budgetDrift, hostInfo.free * 90 / 100);
  EXPECT_LE(allocatorInfo.total, hostInfo.total);

  std::cerr << "Host-unified device memory: " << deviceInfo.free
            << " bytes available; allocator budget: " << allocatorInfo.total
            << " bytes\n";
#endif
}
