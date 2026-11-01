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
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

class GPUDeviceTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      return; // Let individual tests handle errors/skips.

    std::cerr << "##### GPUDeviceTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }
};

// Enabling peer access to self must fail on the active GPU backend.
TEST_F(GPUDeviceTest, EnablePeerAccessSelfErrors) {
  ASSIGN_NO_ERR(DeviceRef, d0, Device::create(Device::gpuAPI, 0));
  // Skip test for Metal
  if (d0->api() == Device::metalAPI) {
    GTEST_SKIP() << "Metal not supported";
  }
  EXPECT_ERR_MSG(d0->enablePeerAccess(*d0), "Cannot enable peer access");
}

} // namespace
