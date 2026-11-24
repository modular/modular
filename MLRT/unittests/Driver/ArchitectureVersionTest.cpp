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

struct Case {
  const char *api;
  const char *name;
  ArchitectureVersion expected;
};

// The names every in-tree backend reports.
const Case kReadable[] = {
    {"cuda", "sm_80", {8, 0, 0}},
    {"cuda", "sm_86", {8, 6, 0}},
    {"cuda", "sm_90a", {9, 0, 0}},
    {"cuda", "sm_100a", {10, 0, 0}},
    {"cuda", "sm_100f", {10, 0, 0}},
    {"cuda", "sm_103a", {10, 3, 0}},
    {"cuda", "sm_120a", {12, 0, 0}},
    {"cuda", "sm_121", {12, 1, 0}},
    {"hip", "gfx90a", {9, 0, 10}},
    {"hip", "gfx942", {9, 4, 2}},
    {"hip", "gfx950", {9, 5, 0}},
    {"hip", "gfx1100", {11, 0, 0}},
    {"hip", "gfx1201", {12, 0, 1}},
    {"hip", "gfx942:sramecc+:xnack-", {9, 4, 2}},
    {"metal", "1", {1, 0, 0}},
    {"metal", "4", {4, 0, 0}},
    {"metal", "5-metal4", {5, 0, 0}},
    {"metal", "12-metal4", {12, 0, 0}},
};

TEST(ArchitectureVersionTest, ReadsEveryBackendsNames) {
  for (const Case &c : kReadable) {
    SCOPED_TRACE(std::string(c.api) + " " + c.name);
    ErrorOr<ArchitectureVersion> version =
        parseArchitectureVersion(c.api, c.name);
    ASSERT_FALSE(version.isError()) << version.getError();
    EXPECT_EQ(version.get(), c.expected);
  }
}

TEST(ArchitectureVersionTest, RejectsNamesItCannotRead) {
  const std::pair<const char *, const char *> unreadable[] = {
      {"cuda", ""},
      {"cuda", "sm_"},
      {"cuda", "sm_9"},
      {"cuda", "sm_90x"},
      {"cuda", "gfx950"},
      {"cuda", "compute_90"},
      {"hip", ""},
      {"hip", "gfx"},
      {"hip", "gfx95"},
      {"hip", "gfx9g0"},
      {"hip", "gfx9x0"},
      {"hip", "gfx9-generic"},
      {"hip", "sm_90a"},
      {"metal", ""},
      {"metal", "apple-m4"},
      {"metal", "5metal4"},
      {"metal", "-metal4"},
      // Target aliases that `--target` passes to a virtual device unchanged.
      // They name no architecture, so they have no version.
      {"hip", "mi355x"},
      {"cuda", "80"},
  };
  for (const auto &[api, name] : unreadable) {
    SCOPED_TRACE(std::string(api) + " '" + name + "'");
    ErrorOr<ArchitectureVersion> version = parseArchitectureVersion(api, name);
    ASSERT_TRUE(version.isError());
    // Quoted, so the check also holds for the empty name.
    const std::string quoted = std::string("'") + name + "'";
    EXPECT_NE(std::string(version.getError()).find(quoted), std::string::npos)
        << version.getError();
  }
}

TEST(ArchitectureVersionTest, RejectsApisWithoutAVersion) {
  EXPECT_TRUE(parseArchitectureVersion("cpu", "").isError());
  // The generic accelerator API names no vendor numbering.
  EXPECT_TRUE(parseArchitectureVersion("gpu", "sm_90a").isError());
  // An API that a driver plugin adds has no numbering this library knows.
  ErrorOr<ArchitectureVersion> plugin =
      parseArchitectureVersion("example_npu", "npu1");
  ASSERT_TRUE(plugin.isError());
  EXPECT_NE(std::string(plugin.getError()).find("example_npu"),
            std::string::npos);
}

TEST(ArchitectureVersionTest, CpuDeviceHasNoVersion) {
  ErrorOr<DeviceRef> cpu = Device::create(Device::cpuAPI, 0);
  if (cpu.isError())
    GTEST_SKIP() << "no CPU device available: " << cpu.getError();
  EXPECT_TRUE(getArchitectureVersion(*cpu.get()).isError());
}

} // namespace
} // namespace M::Driver
