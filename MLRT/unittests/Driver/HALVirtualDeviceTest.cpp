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
// The behavioural contract of a virtual (compile-only) device, exercised
// through the public `Device`/`DeviceContext` API and nothing else.
//
// This target builds in BOTH device-context configurations on purpose. Virtual
// device mode exists twice -- `Virtual/VirtualDeviceContext.cpp` in the legacy
// backend, and the guards in `HAL/HALDeviceContext.cpp` under
// --config=hal-device-context -- so the same assertions run against both and
// the two cannot drift. Assertions the legacy backend pairs with an
// `assert(false)` (it aborts in a debug build where it should return an error)
// are HAL-only, marked below.
//
// No driver plugin and no accelerator are needed: the virtual branch fires
// before any plugin is opened, which is the capability being tested.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>

namespace M::Driver {
namespace {

// The configuration every test runs under. `sm_90a` is a real CUDA target the
// NVPTX compilation device accepts, and deliberately not the arch of any
// machine this test runs on -- it is a cross-compilation target.
constexpr const char *kVirtualAPI = "cuda";
constexpr const char *kVirtualArch = "sm_90a";
constexpr int kVirtualCount = 4;

class HALVirtualDeviceTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Process-global knobs, so every test sets and clears them: a leaked count
    // would leave the rest of the target seeing virtual devices too.
    AsyncRT::setVirtualDeviceAPI(kVirtualAPI);
    AsyncRT::setVirtualDeviceTargetArch(kVirtualArch);
    AsyncRT::setVirtualDeviceCount(kVirtualCount);
  }

  void TearDown() override {
    AsyncRT::setVirtualDeviceCount(0);
    AsyncRT::setVirtualDeviceAPI("");
    AsyncRT::setVirtualDeviceTargetArch("");
  }

  // Failing to build the device is a hard stop rather than a cascade of
  // confusing assertion failures downstream.
  DeviceRef virtualDevice(int id = 0) {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI, id);
    if (device.isError()) {
      ADD_FAILURE() << device.getError();
      return DeviceRef();
    }
    return device.takeValue();
  }
};

// ===-- Identity ---------------------------------------------------------===//

TEST_F(HALVirtualDeviceTest, DeviceIdentity) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);

  // The api is the configured one, not the generic "gpu" that was requested.
  EXPECT_EQ(device->api(), Device::cudaAPI);
  EXPECT_EQ(device->id(), 0);
  EXPECT_EQ(device->name(), "VirtualGPU-0");

  ASSIGN_NO_ERR(const char *, arch, device->getArchName());
  EXPECT_STREQ(arch, kVirtualArch);
}

TEST_F(HALVirtualDeviceTest, DeviceIdCarriesThrough) {
  DeviceRef device = virtualDevice(3);
  ASSERT_NE(device.getPointer(), nullptr);
  EXPECT_EQ(device->id(), 3);
  EXPECT_EQ(device->name(), "VirtualGPU-3");
}

TEST_F(HALVirtualDeviceTest, IsHostUnifiedIsFalseOnVirtualDevices) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(bool, unified, device->isHostUnified());
  EXPECT_FALSE(unified);
}

TEST_F(HALVirtualDeviceTest, NumberOfDevicesReportsVirtualCount) {
  EXPECT_EQ(Device::numberOfDevices(Device::gpuAPI), kVirtualCount);
}

TEST_F(HALVirtualDeviceTest, MissingApiKnobIsRejected) {
  AsyncRT::setVirtualDeviceAPI("");
  EXPECT_ERR_MSG(Device::create(Device::gpuAPI, 0),
                 "Virtual device API not set");
}

TEST_F(HALVirtualDeviceTest, MissingArchKnobIsRejected) {
  AsyncRT::setVirtualDeviceTargetArch("");
  EXPECT_ERR_MSG(Device::create(Device::gpuAPI, 0),
                 "Virtual device target architecture not set");
}

TEST_F(HALVirtualDeviceTest, CpuDeviceIsNotVirtualized) {
  // Virtual mode intercepts GPU apis only; a CPU request must still resolve
  // normally, or a compile-only host loses its host device. The HAL backend
  // reaches a CPU driver plugin to answer this, which a plugin-less lane does
  // not have -- hence the skip rather than a plugin dependency for one case.
  ErrorOr<DeviceRef> device = Device::create(Device::cpuAPI, 0);
  if (device.isError())
    GTEST_SKIP() << "no CPU device available: " << device.getError();
  EXPECT_EQ((*device)->api(), Device::cpuAPI);
  EXPECT_EQ((*device)->name().find("VirtualGPU"), std::string::npos);
}

// ===-- Queries answered truthfully --------------------------------------===//

TEST_F(HALVirtualDeviceTest, IsCompatible) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  EXPECT_NO_ERR(device->isCompatible());
}

TEST_F(HALVirtualDeviceTest, MemoryInfoIsTheSimulatedConstant) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);

  ASSIGN_NO_ERR(Device::MemInfo, info, device->getMemoryInfo());
  EXPECT_EQ(info.total, 192ULL * 1024 * 1024 * 1024);
  EXPECT_EQ(info.free, info.total);

  // Falls out of the above, and is what model-sizing checks read.
  ASSIGN_NO_ERR(size_t, maxAlloc, device->maxSingleAllocationSize());
  EXPECT_EQ(maxAlloc, info.total);
}

TEST_F(HALVirtualDeviceTest, NoPeerAccess) {
  DeviceRef device0 = virtualDevice(0);
  DeviceRef device1 = virtualDevice(1);
  ASSERT_NE(device0.getPointer(), nullptr);
  ASSERT_NE(device1.getPointer(), nullptr);

  ASSIGN_NO_ERR(bool, canAccess, device0->canAccess(*device1));
  EXPECT_FALSE(canAccess);
}

// The api knob takes a `--target-accelerator` vendor name as a synonym for the
// device API it belongs to, so a caller holding one does not have to rewrite
// it (GEX-4019). Asserted on the resolved device rather than the knob: the
// synonym is resolved in `CompilationDevice::create`, and `api()` reporting
// the canonical name is what keeps every `api() == cudaAPI` comparison
// downstream correct.
TEST_F(HALVirtualDeviceTest, VendorNameIsASynonymForTheApi) {
  AsyncRT::setVirtualDeviceAPI("nvidia");
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  EXPECT_EQ(device->api(), Device::cudaAPI);

  ASSIGN_NO_ERR(const char *, arch, device->getArchName());
  EXPECT_STREQ(arch, kVirtualArch);
}

TEST_F(HALVirtualDeviceTest, AmdVendorNamesAreSynonymsForHip) {
  // Only the compilation device is built here: `Device::create` would need the
  // HIP arch in the knob, and the point is the api name, not the arch.
  for (const char *vendor : {"amdgpu", "amd", "hip"}) {
    ASSIGN_NO_ERR(std::unique_ptr<CompilationDevice>, device,
                  CompilationDevice::create(vendor, "gfx942"));
    EXPECT_EQ(device->api(), Device::hipAPI) << vendor;
  }
}

// `DeviceAPINames.h` spells the api names as literals, because the headers
// that declare `Device::cudaAPI` and friends include it. Nothing else makes
// the two agree, so this does -- in both backends, since each declares its own
// copy of the constants.
TEST(DeviceAPINamesTest, MatchesTheDeviceApiConstants) {
  EXPECT_EQ(canonicalDeviceAPI("nvidia"), Device::cudaAPI);
  EXPECT_EQ(canonicalDeviceAPI("amdgpu"), Device::hipAPI);
  EXPECT_EQ(canonicalDeviceAPI("amd"), Device::hipAPI);
  EXPECT_EQ(canonicalDeviceAPI("metal"), Device::metalAPI);

  // Round-trips, so an arch string built from an api parses back to it.
  EXPECT_EQ(canonicalDeviceAPI(apiToVendorPrefix(Device::cudaAPI)),
            Device::cudaAPI);
  EXPECT_EQ(canonicalDeviceAPI(apiToVendorPrefix(Device::hipAPI)),
            Device::hipAPI);
  EXPECT_EQ(canonicalDeviceAPI(apiToVendorPrefix(Device::metalAPI)),
            Device::metalAPI);

  // An api with no distinct vendor name, and a plugin api, pass through.
  EXPECT_EQ(apiToVendorPrefix(Device::metalAPI), Device::metalAPI);
  EXPECT_EQ(apiToVendorPrefix("wombat42"), "wombat42");
  EXPECT_EQ(canonicalDeviceAPI("wombat42"), "wombat42");
}

TEST_F(HALVirtualDeviceTest, UnknownApiNamesTheValuePassedIn) {
  // A vendor name that resolves to nothing must come back verbatim rather than
  // as whatever it was rewritten to, or the message names a string the caller
  // never wrote.
  EXPECT_ERR_MSG(CompilationDevice::create("wombat42", kVirtualArch),
                 "Unknown virtual device API: wombat42");
}

TEST_F(HALVirtualDeviceTest, AcceleratorArchComesFromTheKnobs) {
  // A process-lifetime latch in both backends, so this is the value the whole
  // process sees. A wrong arch here degrades silently into "unknown" at the
  // graph layer and poisons the compile-cache key, which is why it is asserted
  // rather than assumed.
  EXPECT_EQ(Device::getAcceleratorArchOrEmpty(),
            std::string("nvidia:") + kVirtualArch);
}

// ===-- Contexts and streams ---------------------------------------------===//

TEST_F(HALVirtualDeviceTest, ContextCreation) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);

  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  ASSERT_NE(ctx.getPointer(), nullptr);
  EXPECT_EQ(&ctx->device(), device.getPointer());

#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL shim forwards both to the device; the legacy backend answers them
  // on the `Device` only, so there is nothing to compare against there.
  EXPECT_NO_ERR(ctx->isCompatible());
  ASSIGN_NO_ERR(const char *, arch, ctx->getArchName());
  EXPECT_STREQ(arch, kVirtualArch);
#endif
}

TEST_F(HALVirtualDeviceTest, ContextFactoryReachesTheVirtualBranch) {
  ASSIGN_NO_ERR(DeviceContextRef, created,
                DeviceContext::create(Device::gpuAPI, 0));
  EXPECT_EQ(created->device().name(), "VirtualGPU-0");
}

TEST_F(HALVirtualDeviceTest, SynchronizeSucceeds) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Sessions and the graph runtime's device mapping synchronize eagerly while
  // merely setting a model up, so this cannot be an error.
  EXPECT_NO_ERR(ctx->synchronize());
}

TEST_F(HALVirtualDeviceTest, StreamsAreVirtual) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());
  ASSERT_NE(stream.getPointer(), nullptr);
  EXPECT_NO_ERR(stream->synchronize());
  EXPECT_EQ(stream->ptr(), nullptr);

  ASSIGN_NO_ERR(DeviceStreamRef, side, ctx->createStream(0));
  ASSERT_NE(side.getPointer(), nullptr);
  EXPECT_NO_ERR(side->synchronize());
  EXPECT_EQ(side->ptr(), nullptr);
}

TEST_F(HALVirtualDeviceTest, SelectStreamResolvesStreamZeroOnly) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  ASSIGN_NO_ERR(DeviceContextRef, self, ctx->selectStream(0));
  EXPECT_EQ(self.getPointer(), ctx.getPointer());
  EXPECT_ERR(ctx->selectStream(1));
}

// ===-- Operations that must fail loudly ---------------------------------===//

TEST_F(HALVirtualDeviceTest, EventsAreRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  EXPECT_ERR(ctx->enqueueEvent());
  EXPECT_ERR(ctx->eventCreate(/*enableTiming=*/false));
}

TEST_F(HALVirtualDeviceTest, DeviceAttributesAreRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  EXPECT_ERR(device->getAttribute(Device::Attribute::MULTIPROCESSOR_COUNT));
}

TEST_F(HALVirtualDeviceTest, CompletionFlagsAreRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  EXPECT_ERR(ctx->allocateCompletionFlag());
}

// A virtual device has no memory at all, so nothing can be resolved to a host
// address. Both backends report the query as unsupported rather than guessing.
// The HAL backend supports it on no device, so it refuses generically; the
// legacy one supports it on real hardware, so it names the virtual device.
TEST_F(HALVirtualDeviceTest, HostAccessiblePointerIsRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  uint64_t host = 0;
#if MODULAR_HAL_DEVICE_CONTEXT
  EXPECT_ERR_MSG(ctx->getHostAccessiblePointer(&host),
                 "getHostAccessiblePointer is not supported on this device");
#else
  EXPECT_ERR_MSG(ctx->getHostAccessiblePointer(&host),
                 "VirtualDevice does not support host-accessible pointers");
#endif
}

#if MODULAR_HAL_DEVICE_CONTEXT
// Both backends return these errors rather than asserting, so a caller can
// report them; `VirtualDeviceContextTest` pins the same contract on the legacy
// one.

TEST_F(HALVirtualDeviceTest, BufferCreationIsRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Nothing may allocate. This is the invariant that lets DeviceBuffer,
  // DeviceEvent and CompletionFlag carry no virtual-mode guards at all: with no
  // creator able to succeed, no instance can exist on a virtual context.
  EXPECT_ERR(ctx->enqueueCreateBuffer(1024));
  EXPECT_ERR(ctx->enqueueCreateBuffer(0));
  EXPECT_ERR(ctx->createHostBuffer(1024));
  EXPECT_ERR(ctx->createHostBuffer(0));
  EXPECT_ERR(ctx->memmapFile("/dev/null"));

  // These two return a bare ref, so an empty ref is the only refusal available.
  EXPECT_EQ(ctx->createBufferOwning(nullptr, 0, /*owning=*/false).getPointer(),
            nullptr);
  EXPECT_EQ(
      ctx->createHostBufferOwning(nullptr, 0, /*owning=*/false).getPointer(),
      nullptr);
}

TEST_F(HALVirtualDeviceTest, MemoryOperationsAreRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  uint64_t host = 0;
  EXPECT_ERR(ctx->enqueueCopyToDevice(&host, &host, sizeof(host)));
  EXPECT_ERR(ctx->enqueueCopyFromDevice(&host, &host, sizeof(host)));
  EXPECT_ERR(ctx->enqueueSetMemory(&host, sizeof(host), 0, 1));
}

TEST_F(HALVirtualDeviceTest, ThreadBindingAndScopeAreRejected) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  EXPECT_ERR(ctx->bindToCurrentThread());

  DeviceContextScope scope(*ctx);
  EXPECT_ERR(scope.err());
}

TEST_F(HALVirtualDeviceTest, NoDLPackStream) {
  DeviceRef device = virtualDevice();
  ASSERT_NE(device.getPointer(), nullptr);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  EXPECT_FALSE(ctx->getDLPackStream().has_value());
}
#endif // MODULAR_HAL_DEVICE_CONTEXT

} // namespace
} // namespace M::Driver
