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
// Regression test for GEX-3860: MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1 was
// a silent no-op on the real device-initialization path. The device
// MemoryManager used to be built in the base Device constructor, where
// vmmSupported() dispatched to the (always-false) base implementation because
// the derived CUDADevice vtable was not yet installed, so VMM never turned on.
//
// Unlike MemoryManagerVMMTest, which constructs a MemoryManager directly via
// MemoryManager::onDevice() on an already-constructed device (bypassing the
// Device constructor entirely), this test exercises the REAL path: it creates a
// device via Device::create() and then asserts that the device's lazily-built
// memoryManager() VMM state matches the memory_manager_vmm config. Before the
// fix the positive case would fail (VMM silently disabled) even though it was
// requested via the target env and the device supports it. The same source
// backs two targets — one with the env var True, one False — so the negative
// control pins the state to the config rather than to a hardcoded value. These
// tests are single-GPU.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

using namespace M::Driver;
using M::ErrorOr;

namespace {

class MemoryManagerVMMDeviceCreateTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
#if MODULAR_MACOS
    GTEST_SKIP() << "VMM not supported on macOS, skipping VMM tests";
#else
    // Skip macOS before this point so we never do the dead Device::create()
    // work there.
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping VMM tests";
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping VMM tests";
#endif
  }
};

// On a VMM-capable device, the MemoryManager reached through the device created
// by Device::create() must reflect the memory_manager_vmm config: VMM enabled
// when the config requests it, disabled when it does not. This is the path the
// GEX-3860 bug silently disabled — VMM stayed off regardless of the config
// because the manager was built in the base Device constructor before the
// derived vtable was installed. The same source backs two targets (env var True
// and False) so both the positive case (the regression) and the negative
// control (state stays pinned to the config, not hardcoded on) are covered.
TEST_F(MemoryManagerVMMDeviceCreateTest, DeviceCreatePathMatchesVmmConfig) {
  ASSIGN_NO_ERR(DeviceRef, device, Device::create(Device::gpuAPI));
  ASSERT_TRUE(device->vmmSupported());
  const bool expected =
      DeviceContext::configAsBool("memory_manager_vmm", false);
  EXPECT_EQ(device->memoryManager().isVmmEnabled(), expected)
      << "device's MemoryManager VMM state did not match the "
         "memory_manager_vmm "
         "config on the Device::create() path (GEX-3860 regression)";
}

} // namespace
