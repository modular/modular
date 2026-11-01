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
// `Plugin::loadPlugins` reads `MODULAR_DRIVER_PLUGINS` in the format the HAL
// `DeviceContext` reads, so one value serves both. `CPU_PLUGIN` implements only
// the HAL C ABI; `LEGACY_STUB_PLUGIN` implements this loader's entry points.
//
//===----------------------------------------------------------------------===//

#include "Target/DriverPlugin.h"

#include "MLRT/Driver/DeviceContext/DriverPluginSpec.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <string>

using namespace M::Driver;

namespace {

std::string envPath(const char *name) {
  const char *value = std::getenv(name);
  return value ? value : "";
}

int loadPluginsFrom(const std::string &spec, std::string &errors) {
  setenv("MODULAR_DRIVER_PLUGINS", spec.c_str(), /*overwrite=*/1);
  testing::internal::CaptureStderr();
  auto loaded = Plugin::loadPlugins();
  errors = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(loaded.isError());
  return loaded.isError() ? -1 : loaded.get();
}

} // namespace

TEST(DriverPluginLoadTest, LoadsFromNamedEntriesAndSkipsHALOnlyPlugins) {
  std::string errors;
  EXPECT_EQ(loadPluginsFrom("CPU@" + envPath("CPU_PLUGIN") + ";Stub@" +
                                envPath("LEGACY_STUB_PLUGIN"),
                            errors),
            1);
  EXPECT_TRUE(Plugin::hasPluginForApi("legacystub"));
  EXPECT_EQ(errors, "");
}

TEST(DriverPluginLoadTest, LoadsFromABarePath) {
  std::string errors;
  EXPECT_EQ(loadPluginsFrom(envPath("LEGACY_STUB_PLUGIN"), errors), 1);
  EXPECT_TRUE(Plugin::hasPluginForApi("legacystub"));
  EXPECT_EQ(errors, "");
}

TEST(DriverPluginLoadTest, ReportsAPluginThatCannotBeOpened) {
  std::string errors;
  EXPECT_EQ(loadPluginsFrom("Missing@/nonexistent/plugin.so", errors), 0);
  EXPECT_NE(errors.find("Failed to load driver plugin"), std::string::npos)
      << errors;
}

TEST(DriverPluginLoadTest, ParsesNamedAndBareEntries) {
  auto plugins = parseDriverPluginSpec("CUDA@/a/cuda.so;;/b/stub.so;Empty@");
  ASSERT_EQ(plugins.size(), 2u);
  EXPECT_EQ(plugins[0].name, "CUDA");
  EXPECT_EQ(plugins[0].path, "/a/cuda.so");
  EXPECT_EQ(plugins[1].name, "");
  EXPECT_EQ(plugins[1].path, "/b/stub.so");
}
