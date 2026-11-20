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
// The local accelerator arch the compiler family reads, from the accelerator
// plugin `MODULAR_DRIVER_PLUGINS` names or, without one, the vendor libraries.
//
// `MODULAR_DRIVER_PLUGINS` lists the CPU plugin ahead of the accelerator's, so
// the query has to skip the CPU entry, by its name or, for a bare path, by the
// api the plugin reports.
// `EXPECTED_ARCH` names the exact arch the accelerator should report, and
// `EXPECTED_ARCH_PREFIX` its vendor prefix. `UNEXPECTED_ARCH_PREFIX` names a
// vendor the query must not report, as for a bare-path plugin whose api has no
// Mojo vendor: the query skips it and asks the vendor libraries, which find
// the host's own accelerator, if any. Without any of these, no arch is
// expected.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>
#include <string>

using M::Driver::Device;

TEST(AcceleratorArchTest, ReadsTheAcceleratorArch) {
  const std::string &arch = Device::getAcceleratorArchOrEmpty();
  // Logged so the plugin and vendor-library runs on one GPU can be compared.
  std::cout << "arch: \"" << arch << "\"\n";
  if (const char *expected = std::getenv("EXPECTED_ARCH")) {
    EXPECT_EQ(arch, expected);
    return;
  }
  if (const char *unexpected = std::getenv("UNEXPECTED_ARCH_PREFIX")) {
    EXPECT_FALSE(llvm::StringRef(arch).starts_with(unexpected))
        << "arch: \"" << arch << "\"";
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
