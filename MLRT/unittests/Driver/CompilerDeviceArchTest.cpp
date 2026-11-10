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
// The local accelerator arch the compiler family reads, through
// `CompilerDeviceArch.cpp` under --config=hal-device-context.
//
// `MODULAR_DRIVER_PLUGINS` lists the CPU plugin ahead of the accelerator's, so
// the query has to skip the CPU entry, by its name or, for a bare path, by the
// api the plugin reports.
// `EXPECTED_ARCH` names the exact arch the accelerator should report, and
// `EXPECTED_ARCH_PREFIX` its vendor prefix; without either, no arch is
// expected, as for a bare-path plugin whose api has no Mojo vendor.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <string>

using M::Driver::Device;

TEST(CompilerDeviceArchTest, ReadsTheAcceleratorPluginArch) {
  const std::string &arch = Device::getAcceleratorArchOrEmpty();
  if (const char *expected = std::getenv("EXPECTED_ARCH")) {
    EXPECT_EQ(arch, expected);
    return;
  }
  const char *expectedPrefix = std::getenv("EXPECTED_ARCH_PREFIX");
  if (!expectedPrefix) {
    EXPECT_EQ(arch, "");
    return;
  }
  EXPECT_TRUE(llvm::StringRef(arch).starts_with(expectedPrefix))
      << "arch: \"" << arch << "\"";
  EXPECT_GT(arch.size(), std::string(expectedPrefix).size());
}
