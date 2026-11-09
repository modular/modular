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
// A HAL device reports the api its plugin names, as the legacy driver's plugin
// backends do, even when the HAL has no constant for it. Kernel registrations
// select on that api, so a device that reported the generic "gpu" would run
// generic GPU kernels in place of its own.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "Support/ErrorOr.h"

#include "gtest/gtest.h"

using M::ErrorOr;
using M::Driver::Device;
using M::Driver::DeviceRef;

TEST(HALPluginAPITest, DeviceReportsThePluginsOwnAPI) {
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  ASSERT_FALSE(device.isError()) << device.getError();
  EXPECT_EQ((*device)->api().str(), "mock");
}
