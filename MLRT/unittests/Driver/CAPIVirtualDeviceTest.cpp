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
// The driver C API on a virtual (compile-only) device, which needs no GPU.
//
// Virtual mode is process-global and the C API caches one device context per
// ordinal, so this binary configures one target for the whole process.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "MLRT/Driver/CAPI/DriverCAPI.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <string>

namespace {

// `--target amdgpu:mi355x` hands the virtual device this alias unchanged.
// TODO(DRIV-717): once the alias resolves to `gfx950` where the virtual device
// is configured, assert that name and its version here instead.
constexpr const char *kAliasTarget = "mi355x";

std::string lastError() {
  const size_t size = M_DriverLastErrorMessage(nullptr, 0);
  std::string message(size, '\0');
  M_DriverLastErrorMessage(message.data(), size);
  message.resize(size == 0 ? 0 : size - 1);
  return message;
}

class CAPIVirtualDeviceTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    M::AsyncRT::setVirtualDeviceAPI("hip");
    M::AsyncRT::setVirtualDeviceTargetArch(kAliasTarget);
    M::AsyncRT::setVirtualDeviceCount(1);
  }
};

// A name the library cannot read is a gap in the library, not a driver
// failure, so it is `NOT_SUPPORTED` and the message includes the name.
TEST_F(CAPIVirtualDeviceTest, UnreadableArchitectureNameIsNotSupported) {
  M_DriverDevice *device = nullptr;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, &device), M_DRIVER_OK)
      << lastError();

  char arch[64] = {};
  ASSERT_GT(M_DriverDeviceArchitectureName(device, arch, sizeof(arch)), 1u)
      << lastError();
  ASSERT_STREQ(arch, kAliasTarget);

  uint32_t major = 1234;
  EXPECT_EQ(M_DriverDeviceArchitectureVersion(device, &major, nullptr, nullptr),
            M_DRIVER_NOT_SUPPORTED);
  EXPECT_EQ(major, 1234u);
  EXPECT_NE(lastError().find(kAliasTarget), std::string::npos) << lastError();

  EXPECT_EQ(M_DriverDeviceDestroy(device), M_DRIVER_OK);
}

} // namespace
