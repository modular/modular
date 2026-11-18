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
// A plugin refusal reaches the HAL DeviceContext's caller as an error, not as
// a process abort. The mock plugin refuses the entry point that
// MOCK_FAIL_CALL names.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "ScopedEnvVar.h"
#include "Support/ErrorOr.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

using M::ErrorOr;
using M::ErrorOrSuccess;
using M::Driver::Device;
using M::Driver::DeviceBufferRef;
using M::Driver::DeviceContextRef;
using M::Driver::DeviceRef;

namespace {

/// Makes the mock plugin refuse `entryPoint` for this object's lifetime.
class MockRefusal : ScopedEnvVar {
public:
  explicit MockRefusal(const char *entryPoint)
      : ScopedEnvVar("MOCK_FAIL_CALL", entryPoint) {}
};

DeviceContextRef createContext(int id = 0) {
  ErrorOr<DeviceContextRef> ctx =
      M::Driver::DeviceContext::create(Device::gpuAPI, id);
  EXPECT_FALSE(ctx.isError()) << ctx.getError();
  return ctx.takeValue();
}

} // namespace

TEST(HALBridgeErrorTest, RefusedWrapReturnsAnEmptyRef) {
  DeviceContextRef ctx = createContext();
  std::vector<char> memory(64);
  MockRefusal refusal("M_driver_memory_wrap");
  EXPECT_FALSE(ctx->createBufferOwning(memory.data(), memory.size(),
                                       /*owning=*/false));
  EXPECT_FALSE(ctx->createHostBufferOwning(memory.data(), memory.size(),
                                           /*owning=*/false));
}

TEST(HALBridgeErrorTest, RefusedSubBufferIsAnError) {
  DeviceContextRef ctx = createContext();
  ErrorOr<DeviceBufferRef> device = ctx->enqueueCreateBuffer(64);
  ASSERT_FALSE(device.isError()) << device.getError();
  ErrorOr<DeviceBufferRef> host = ctx->createHostBuffer(64);
  ASSERT_FALSE(host.isError()) << host.getError();

  MockRefusal refusal("M_driver_memory_wrap");
  EXPECT_TRUE((*device)->createSubBuffer(0, 16).isError());
  EXPECT_TRUE((*host)->createSubBuffer(0, 16).isError());
}

TEST(HALBridgeErrorTest, RefusedCopyAndMemsetAreErrors) {
  DeviceContextRef ctx = createContext();
  ErrorOr<DeviceBufferRef> device = ctx->enqueueCreateBuffer(64);
  ASSERT_FALSE(device.isError()) << device.getError();
  std::vector<char> host(64);

  MockRefusal refusal("M_driver_memory_wrap");
  EXPECT_TRUE(
      ctx->enqueueCopyToDevice((*device)->ptr(), host.data(), host.size())
          .isError());
  EXPECT_TRUE(
      ctx->enqueueCopyFromDevice(host.data(), (*device)->ptr(), host.size())
          .isError());
  EXPECT_TRUE(
      ctx->enqueueSetMemory((*device)->ptr(), host.size(), 0, 1).isError());
}

TEST(HALBridgeErrorTest, RefusedMemmapIsAnError) {
  DeviceContextRef ctx = createContext();
  std::filesystem::path path =
      std::filesystem::path(testing::TempDir()) / "memmap.bin";
  std::ofstream(path) << "contents";

  MockRefusal refusal("M_driver_memory_wrap");
  EXPECT_TRUE(ctx->memmapFile(path).isError());
}

// Destroying an owning buffer claims the memory from the plugin. A refusal
// leaks the memory rather than aborting in the destructor.
TEST(HALBridgeErrorTest, RefusedClaimAtDestructionIsNotFatal) {
  DeviceContextRef ctx = createContext();
  void *memory = std::malloc(64);
  DeviceBufferRef buffer = ctx->createBufferOwning(memory, 64,
                                                   /*owning=*/true);
  ASSERT_TRUE(buffer);
  {
    MockRefusal refusal("M_driver_memory_wrap");
    buffer = DeviceBufferRef();
  }
  std::free(memory);
}

TEST(HALBridgeErrorTest, RefusedReleaseKeepsOwnership) {
  DeviceContextRef ctx = createContext();
  ErrorOr<DeviceBufferRef> buffer = ctx->enqueueCreateBuffer(64);
  ASSERT_FALSE(buffer.isError()) << buffer.getError();
  void *ptr = (*buffer)->ptr();

  MockRefusal refusal("M_driver_memory_unwrap");
  EXPECT_EQ((*buffer)->releasePtr(), nullptr);
  EXPECT_EQ((*buffer)->ptr(), ptr);
}

TEST(HALBridgeErrorTest, RefusedContextIsAnError) {
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 0);
  ASSERT_FALSE(device.isError()) << device.getError();

  MockRefusal refusal("M_driver_context_create");
  EXPECT_TRUE((*device)->createContext().isError());
}

// Device 1, so that contexts the other tests hold on device 0 don't count
// against the limit.
TEST(HALBridgeErrorTest, OneContextPerDeviceIsShared) {
  ScopedEnvVar limit("MOCK_MAX_CONTEXTS_PER_DEVICE", "1");
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, 1);
  ASSERT_FALSE(device.isError()) << device.getError();
  ErrorOr<DeviceContextRef> first = (*device)->createContext();
  ASSERT_FALSE(first.isError()) << first.getError();
  ErrorOr<DeviceContextRef> second = (*device)->createContext();
  ASSERT_FALSE(second.isError()) << second.getError();
}
