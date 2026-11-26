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
// The device count must match what CUDA can open. NVML ignores
// `CUDA_VISIBLE_DEVICES`, so this test narrows it to one GPU on a multi-GPU
// node before the plugin initializes CUDA. It lives in its own binary because
// `cuInit` reads the variable once per process.
//
//===----------------------------------------------------------------------===//

#include "Fixture.h"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <string>

namespace {

struct CUDAVisibleDevicesTest : DriverPluginTest {
  void SetUp() override {
    std::string visible = "0";
    if (const char *env = std::getenv("CUDA_VISIBLE_DEVICES"); env && *env)
      visible = llvm::StringRef(env).split(',').first.str();
    ASSERT_EQ(setenv("CUDA_VISIBLE_DEVICES", visible.c_str(), 1), 0);
    DriverPluginTest::SetUp();
  }
};

TEST_F(CUDAVisibleDevicesTest, CountsOnlyVisibleDevices) {
  int64_t deviceCount = 0;
  ASSERT_SUCCESS(device_count(driver, &deviceCount));
  EXPECT_EQ(deviceCount, 1);
  for (int64_t i = 0; i < deviceCount; ++i) {
    M_driver_device_t device;
    EXPECT_SUCCESS(device_get(driver, i, &device));
  }
}

} // namespace
