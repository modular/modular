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
// HAL DeviceContexts compare equal when they drive the same stream, as legacy
// ones do.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "ScopedEnvVar.h"
#include "Support/ErrorOr.h"

#include "gtest/gtest.h"

using M::ErrorOr;
using M::Driver::Device;
using M::Driver::DeviceContext;
using M::Driver::DeviceContextRef;
using M::Driver::DeviceRef;

TEST(HALContextEqualityTest, BaseStreamViewEqualsItsContext) {
  ErrorOr<DeviceContextRef> ctx = DeviceContext::create(Device::gpuAPI, 0);
  ASSERT_FALSE(ctx.isError()) << ctx.getError();
  ErrorOr<DeviceContextRef> view = (*ctx)->selectStream(0);
  ASSERT_FALSE(view.isError()) << view.getError();
  EXPECT_TRUE(**view == **ctx);

  ASSERT_FALSE((*ctx)->createStream().isError());
  ErrorOr<DeviceContextRef> side = (*ctx)->selectStream(1);
  ASSERT_FALSE(side.isError()) << side.getError();
  EXPECT_FALSE(**side == **ctx);
}

TEST(HALContextEqualityTest, ContextsOnSeparateStreamsDiffer) {
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  ASSERT_FALSE(device.isError()) << device.getError();
  ErrorOr<DeviceContextRef> first = (*device)->createContext();
  ASSERT_FALSE(first.isError()) << first.getError();
  ErrorOr<DeviceContextRef> second = (*device)->createContext();
  ASSERT_FALSE(second.isError()) << second.getError();
  EXPECT_FALSE(**first == **second);
}

// A device that allows one context shares it, stream included.
TEST(HALContextEqualityTest, SharedContextsAreEqual) {
  ScopedEnvVar limit("MOCK_MAX_CONTEXTS_PER_DEVICE", "1");
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 1);
  ASSERT_FALSE(device.isError()) << device.getError();
  ErrorOr<DeviceContextRef> first = (*device)->createContext();
  ASSERT_FALSE(first.isError()) << first.getError();
  ErrorOr<DeviceContextRef> second = (*device)->createContext();
  ASSERT_FALSE(second.isError()) << second.getError();
  EXPECT_TRUE(**first == **second);
}
