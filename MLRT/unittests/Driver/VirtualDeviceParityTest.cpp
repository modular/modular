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
// A virtual (compile-only) device stands in for the real one when a graph is
// compiled without the target accelerator. Graph code and kernel selection
// branch on the device's API, architecture name and architecture version, so
// a virtual device configured for a real device's target must report the same
// three.
//
// The test configures the virtual device with the real device's own API and
// architecture name. It does not check that tools passing `--target` spell
// the target the way the device does.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "MLRT/Driver/CAPI/ArchitectureVersion.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

#include <ostream>
#include <string>

namespace M::Driver {

void PrintTo(const ArchitectureVersion &v, std::ostream *os) {
  *os << v.major << "." << v.minor << "." << v.stepping;
}

namespace {

/// Virtual device mode is process-global, so restore it on scope exit.
class VirtualTargetScope {
public:
  VirtualTargetScope(llvm::StringRef api, llvm::StringRef arch) {
    AsyncRT::setVirtualDeviceAPI(api);
    AsyncRT::setVirtualDeviceTargetArch(arch);
    AsyncRT::setVirtualDeviceCount(1);
  }
  ~VirtualTargetScope() {
    AsyncRT::setVirtualDeviceCount(0);
    AsyncRT::setVirtualDeviceAPI("");
    AsyncRT::setVirtualDeviceTargetArch("");
  }
};

TEST(VirtualDeviceParityTest, VirtualDeviceReportsTheRealDevicesIdentity) {
  // The target requires a GPU, so failing here is a broken lane (for example
  // a HAL build with no accelerator plugin), not a skip.
  ErrorOr<DeviceRef> realOr = Device::create(Device::gpuAPI, 0);
  ASSERT_FALSE(realOr.isError()) << realOr.getError();
  DeviceRef real = realOr.takeValue();
  if (real->api() == Device::metalAPI)
    GTEST_SKIP() << "Metal has no virtual compilation device (MOCO-2411)";

  ErrorOr<const char *> realArch = real->getArchName();
  ASSERT_FALSE(realArch.isError()) << realArch.getError();
  const std::string arch = realArch.get();
  ErrorOr<ArchitectureVersion> realVersion = getArchitectureVersion(*real);
  ASSERT_FALSE(realVersion.isError()) << realVersion.getError();

  VirtualTargetScope scope(real->api(), arch);
  ErrorOr<DeviceRef> virtOr = Device::create(Device::gpuAPI, 0);
  ASSERT_FALSE(virtOr.isError()) << virtOr.getError();
  DeviceRef virt = virtOr.takeValue();
  // The HAL build makes a new real device on every call, so only the name
  // shows that virtual mode took effect.
  ASSERT_EQ(virt->name(), "VirtualGPU-0")
      << "virtual device mode did not take effect";

  EXPECT_EQ(virt->api(), real->api());
  ErrorOr<const char *> virtArch = virt->getArchName();
  ASSERT_FALSE(virtArch.isError()) << virtArch.getError();
  EXPECT_EQ(std::string(virtArch.get()), arch);
  ErrorOr<ArchitectureVersion> virtVersion = getArchitectureVersion(*virt);
  ASSERT_FALSE(virtVersion.isError()) << virtVersion.getError();
  EXPECT_EQ(virtVersion.get(), realVersion.get());

  // The model name is not compared: `sm_100a` covers both B200 and GB200, so
  // a virtual device cannot know the model. Kernel selection that matches on
  // the model therefore differs in a virtual compile.
}

} // namespace
} // namespace M::Driver
