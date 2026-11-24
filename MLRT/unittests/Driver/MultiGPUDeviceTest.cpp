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

#include "llvm/Support/Process.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

class MultiGPUDeviceTest : public ::testing::Test {};

// Takes the HIP runtime's per-thread last error, or 0 off HIP. Looked up in the
// library the driver already loaded, which is what any other HIP user in the
// process, torch for one, checks after its own kernel launches.
int takeHipLastError(const Device &device) {
  if (device.api() != Device::hipAPI)
    return 0;
  void *lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
  if (lib == nullptr)
    return -1;
  auto *getLastError =
      reinterpret_cast<int (*)()>(dlsym(lib, "hipGetLastError"));
  int err = getLastError ? getLastError() : -1;
  dlclose(lib);
  return err;
}

TEST_F(MultiGPUDeviceTest, DeviceVisibilityControl) {
  std::string const &arch = Device::getAcceleratorArchOrEmpty();
  const char *envVarName = nullptr;
  if (arch.find("nvidia") != std::string::npos) {
    envVarName = "CUDA_VISIBLE_DEVICES";
  } else if (arch.find("amd") != std::string::npos) {
    // Multiple options for AMD, we only support one
    // https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
    envVarName = "HIP_VISIBLE_DEVICES";
  } else {
    return;
  }
  std::optional<std::string> prevEnvVarValue =
      llvm::sys::Process::GetEnv(envVarName);
  unsetenv(envVarName);
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 1) << "Should only run on multi-gpu machines";
  int const allDeviceCount = num;

  setenv(envVarName, "0", 1 /*overwrite*/);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, 1);

  setenv(envVarName, "1,0", 1 /*overwrite*/);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, 2);

  setenv(envVarName, "", 1 /*overwrite*/);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, 0) << "Empty string => 0 devices as per CUDA specs";
  // ... we extend this to HIP too

  // Double check
  unsetenv(envVarName);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, allDeviceCount);

  // Restore original setting
  if (prevEnvVarValue) {
    setenv(envVarName, prevEnvVarValue->c_str(), 1 /*overwrite*/);
  }
}

// EnablePeerAccess is idempotent for the active GPU backend (CUDA/HIP/Metal).
TEST_F(MultiGPUDeviceTest, EnablePeerAccessIdempotent) {
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 1) << "Should only run on multi-gpu machines";

  bool foundPair = false;
  for (int i = 0; i < num && !foundPair; ++i) {
    ASSIGN_NO_ERR(DeviceRef, di, Device::create(Device::gpuAPI, i));
    for (int j = i + 1; j < num && !foundPair; ++j) {
      ASSIGN_NO_ERR(DeviceRef, dj, Device::create(Device::gpuAPI, j));

      ASSIGN_NO_ERR(bool, canIJ, di->canAccess(*dj));
      ASSIGN_NO_ERR(bool, canJI, dj->canAccess(*di));
      if (!canIJ || !canJI)
        continue;

      EXPECT_NO_ERR(di->enablePeerAccess(*dj));
      EXPECT_NO_ERR(dj->enablePeerAccess(*di));
      // Idempotency: repeat should also succeed.
      EXPECT_NO_ERR(di->enablePeerAccess(*dj));
      EXPECT_NO_ERR(dj->enablePeerAccess(*di));
      // HIP reports the repeat as hipErrorPeerAccessAlreadyEnabled, which
      // must not stay behind as the last error once it has been tolerated.
      EXPECT_EQ(takeHipLastError(*di), 0);

      foundPair = true;
    }
  }

  if (!foundPair) {
    GTEST_SKIP() << "No GPU device pair with mutual P2P access; skipping.";
  }
}

} // namespace
