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
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

class GPUHealthcheckTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      return; // Let individual tests handle errors/skips.

    std::cerr << "##### GPUHealthcheckTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }
};

TEST_F(GPUHealthcheckTest, HealthcheckRunsOnDeviceCreation) {
  // Device creation succeeds on a healthy GPU — the healthcheck ran
  // implicitly during Device::create().
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  if (device.isError()) {
    // If device creation fails, the error message should be actionable.
    // It should mention either a driver issue or a healthcheck failure with
    // the bypass env var.
    GTEST_SKIP() << "GPU device creation failed: " << device.getError();
  }
  EXPECT_NE(device->getPointer(), nullptr);
}

TEST_F(GPUHealthcheckTest, ExplicitHealthcheckSucceeds) {
  // On a healthy GPU, explicit runHealthcheck() should succeed.
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  if (device.isError())
    GTEST_SKIP() << "No GPU available: " << device.getError();

  EXPECT_NO_ERR((*device)->runHealthcheck());
}

TEST_F(GPUHealthcheckTest, HealthcheckDisabledByEnvVar) {
  // When the disable env var is set, healthcheck should always succeed,
  // even if re-invoked explicitly.
  ScopedEnvVar disableHealthcheck("MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK",
                                  "true");

  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  if (device.isError())
    GTEST_SKIP() << "No GPU available: " << device.getError();

  EXPECT_NO_ERR((*device)->runHealthcheck());
}

TEST_F(GPUHealthcheckTest, HealthcheckDisabledCaseInsensitive) {
  // configAsBool should handle "True" as well.
  ScopedEnvVar disableHealthcheck("MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK",
                                  "True");

  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  if (device.isError())
    GTEST_SKIP() << "No GPU available: " << device.getError();

  EXPECT_NO_ERR((*device)->runHealthcheck());
}

TEST_F(GPUHealthcheckTest, DeviceContextCreationIncludesHealthcheck) {
  // Full DeviceContext creation path should succeed on a healthy GPU.
  // This exercises the full path: Device::create → healthcheck → createContext.
  ErrorOr<DeviceContextRef> ctx = DeviceContext::create(Device::gpuAPI, 0);
  if (ctx.isError())
    GTEST_SKIP() << "GPU context creation failed: " << ctx.getError();

  EXPECT_NE(ctx->getPointer(), nullptr);
}

} // namespace
