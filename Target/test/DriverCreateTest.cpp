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
// `createDriver` against in-process stand-ins for a plugin's `M_driver_create`,
// and the legacy loader's use of it against `LEGACY_STUB_PLUGIN`. A separate
// binary from `DriverPluginLoadTest`, because the loader keeps every plugin it
// loads for the life of the process.
//
//===----------------------------------------------------------------------===//

#include "Target/DriverCreate.h"
#include "Target/DriverPlugin.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <string>

using namespace M::Driver::Plugin;

namespace {

int driverHandle;
int failureHandle;
int destroyedHandles;
M_driver_version pluginVersion;
M_driver_status_t createStatus;
bool returnFailureHandle;

M_driver_status_t fakeCreate(M_driver_version *version,
                             M_driver_driver_t *driver) {
  *version = pluginVersion;
  if (createStatus == M_DRIVER_STATUS_SUCCESS)
    *driver = reinterpret_cast<M_driver_driver_t>(&driverHandle);
  else if (returnFailureHandle)
    *driver = reinterpret_cast<M_driver_driver_t>(&failureHandle);
  return createStatus;
}

M_driver_status_t fakeDestroy(M_driver_driver_t) {
  ++destroyedHandles;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t fakeStatusMessage(M_driver_driver_t driver, M_driver_status_t,
                                    char *buffer, int64_t size) {
  if (driver != reinterpret_cast<M_driver_driver_t>(&failureHandle))
    return M_DRIVER_STATUS_BAD_DRIVER;
  std::strncpy(buffer, "no accelerator runtime", size - 1);
  return M_DRIVER_STATUS_SUCCESS;
}

class DriverCreateTest : public testing::Test {
protected:
  void SetUp() override {
    destroyedHandles = 0;
    pluginVersion = hostDriverVersion;
    createStatus = M_DRIVER_STATUS_SUCCESS;
    returnFailureHandle = false;
  }

  M::ErrorOr<M_driver_driver_t> create() {
    return createDriver("/plugins/fake.so", fakeCreate, fakeDestroy,
                        fakeStatusMessage);
  }
};

std::string versionString(uint32_t major, uint32_t minor) {
  return std::to_string(major) + "." + std::to_string(minor) + ".";
}

} // namespace

TEST(DriverVersionTest, RequiresTheSameMajorAndAtLeastTheHostMinor) {
  M_driver_version host = {1, 2, 3};
  EXPECT_TRUE(isCompatibleDriverVersion(host, {1, 2, 0}));
  EXPECT_TRUE(isCompatibleDriverVersion(host, {1, 3, 0}));
  EXPECT_FALSE(isCompatibleDriverVersion(host, {1, 1, 9}));
  EXPECT_FALSE(isCompatibleDriverVersion(host, {0, 2, 3}));
  EXPECT_FALSE(isCompatibleDriverVersion(host, {2, 2, 3}));
}

TEST_F(DriverCreateTest, AcceptsTheHostVersion) {
  auto driver = create();
  ASSERT_FALSE(driver.isError()) << driver.getError();
  EXPECT_EQ(driver.get(), reinterpret_cast<M_driver_driver_t>(&driverHandle));
  EXPECT_EQ(destroyedHandles, 0);
}

TEST_F(DriverCreateTest, AcceptsANewerMinorVersion) {
  pluginVersion.minor += 1;
  EXPECT_FALSE(create().isError());
}

TEST_F(DriverCreateTest, RejectsAnOlderMinorVersionAndDestroysTheDriver) {
  if (hostDriverVersion.minor == 0)
    GTEST_SKIP() << "no older minor version exists";
  pluginVersion.minor = hostDriverVersion.minor - 1;
  auto driver = create();
  ASSERT_TRUE(driver.isError());
  std::string error = driver.getError();
  EXPECT_NE(error.find("/plugins/fake.so"), std::string::npos) << error;
  EXPECT_NE(error.find(versionString(pluginVersion.major, pluginVersion.minor)),
            std::string::npos)
      << error;
  EXPECT_NE(error.find(versionString(hostDriverVersion.major,
                                     hostDriverVersion.minor)),
            std::string::npos)
      << error;
  EXPECT_EQ(destroyedHandles, 1);
}

TEST_F(DriverCreateTest, RejectsAnotherMajorVersion) {
  pluginVersion.major += 1;
  createStatus = M_DRIVER_STATUS_INCOMPATIBLE_VERSION;
  auto driver = create();
  ASSERT_TRUE(driver.isError());
  std::string error = driver.getError();
  EXPECT_NE(error.find("incompatible"), std::string::npos) << error;
  EXPECT_NE(error.find(versionString(pluginVersion.major, pluginVersion.minor)),
            std::string::npos)
      << error;
  EXPECT_EQ(destroyedHandles, 0);
}

// The fake leaves the host's version in place, as a plugin that does not write
// its version back would.
TEST_F(DriverCreateTest, SaysWhenARefusingPluginReportsNoVersion) {
  createStatus = M_DRIVER_STATUS_INCOMPATIBLE_VERSION;
  auto driver = create();
  ASSERT_TRUE(driver.isError());
  std::string error = driver.getError();
  EXPECT_NE(error.find("without reporting the version it implements"),
            std::string::npos)
      << error;
  EXPECT_NE(error.find("/plugins/fake.so"), std::string::npos) << error;
}

// A plugin must refuse another major version itself; one that does not is
// still rejected.
TEST_F(DriverCreateTest, RejectsAnotherMajorVersionThePluginAccepted) {
  pluginVersion.major += 1;
  auto driver = create();
  ASSERT_TRUE(driver.isError());
  EXPECT_EQ(destroyedHandles, 1);
}

TEST_F(DriverCreateTest, ReportsTheMessageOfAFailureHandleAndDestroysIt) {
  createStatus = M_DRIVER_STATUS_CUSTOM_CODES_BASE;
  returnFailureHandle = true;
  auto driver = create();
  ASSERT_TRUE(driver.isError());
  std::string error = driver.getError();
  EXPECT_NE(error.find("no accelerator runtime"), std::string::npos) << error;
  EXPECT_NE(error.find("/plugins/fake.so"), std::string::npos) << error;
  EXPECT_EQ(destroyedHandles, 1);
}

TEST_F(DriverCreateTest, TreatsNoDevicesAsAFailure) {
  createStatus = M_DRIVER_STATUS_NO_DEVICES;
  auto driver = create();
  ASSERT_TRUE(driver.isError());
  std::string error = driver.getError();
  EXPECT_NE(error.find(std::to_string(M_DRIVER_STATUS_NO_DEVICES)),
            std::string::npos)
      << error;
  EXPECT_EQ(destroyedHandles, 0);
}

TEST(DriverPluginLoaderVersionTest, RejectsAPluginWithAnOlderMinorVersion) {
  if (hostDriverVersion.minor == 0)
    GTEST_SKIP() << "no older minor version exists";
  std::string minor = std::to_string(hostDriverVersion.minor - 1);
  setenv("LEGACY_STUB_VERSION_MINOR", minor.c_str(), /*overwrite=*/1);
  setenv("MODULAR_DRIVER_PLUGINS", std::getenv("LEGACY_STUB_PLUGIN"),
         /*overwrite=*/1);
  testing::internal::CaptureStderr();
  auto loaded = loadPlugins();
  std::string errors = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(loaded.isError());
  EXPECT_EQ(loaded.get(), 0);
  EXPECT_FALSE(hasPluginForApi("legacystub"));
  EXPECT_NE(errors.find("incompatible"), std::string::npos) << errors;
}
