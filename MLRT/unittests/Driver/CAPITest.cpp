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
// Tests for the driver C API (MLRT/Driver/CAPI/DriverCAPI.h).
//
// The CPU device is always available, so the device, stream, buffer and copy
// tests run everywhere, including on a machine with no accelerator. Tests that
// need an accelerator skip when there is none, so this file is meaningful on a
// developer workstation and on a GPU host alike.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/CAPI/DriverCAPI.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "CAPITestHelpers.h"

#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <dlfcn.h>

// The build defines this only under the HAL device context, and the test is
// compiled with every undefined macro in an #if treated as an error, so the
// classic build spells out the zero.
#ifndef MODULAR_HAL_DEVICE_CONTEXT
#define MODULAR_HAL_DEVICE_CONTEXT 0
#endif

#if !MODULAR_HAL_DEVICE_CONTEXT
#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#endif

using namespace capi_test;

namespace {

/// The vendor's creation flags for `stream` (`cuStreamGetFlags` /
/// `hipStreamGetFlags`); nullopt on a backend without them. The driver has
/// already loaded the vendor library, so RTLD_NOLOAD finds it.
std::optional<unsigned> vendorStreamFlags(const M_DriverDevice *device,
                                          const M_DriverStream *stream) {
  char api[32] = {};
  if (M_DriverDeviceApi(device, api, sizeof(api)) == 0)
    return std::nullopt;
  const char *library = nullptr;
  const char *symbol = nullptr;
  if (std::strcmp(api, "cuda") == 0) {
    library = "libcuda.so.1";
    symbol = "cuStreamGetFlags";
  } else if (std::strcmp(api, "hip") == 0) {
    library = "libamdhip64.so";
    symbol = "hipStreamGetFlags";
  } else {
    return std::nullopt;
  }
  void *handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    ADD_FAILURE() << library << " is not loaded";
    return std::nullopt;
  }
  using GetFlags = int (*)(void *, unsigned *);
  auto getFlags = reinterpret_cast<GetFlags>(dlsym(handle, symbol));
  void *native = nullptr;
  unsigned flags = 0;
  const bool ok = getFlags != nullptr &&
                  M_DriverStreamNativeHandle(stream, &native) == M_DRIVER_OK &&
                  native != nullptr && getFlags(native, &flags) == 0;
  dlclose(handle);
  if (!ok) {
    ADD_FAILURE() << symbol << " did not answer for the stream";
    return std::nullopt;
  }
  return flags;
}

} // namespace

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

TEST(DriverCApi, VersionMatchesTheHeader) {
  const uint32_t version = M_DriverCApiVersion();
  EXPECT_EQ(version,
            (static_cast<uint32_t>(M_DRIVER_CAPI_VERSION_MAJOR) << 24) |
                (static_cast<uint32_t>(M_DRIVER_CAPI_VERSION_MINOR) << 16) |
                static_cast<uint32_t>(M_DRIVER_CAPI_VERSION_PATCH));
  EXPECT_EQ(version >> 24, 1u);
  EXPECT_EQ((version >> 16) & 0xffu, 3u);
  EXPECT_EQ(version & 0xffffu, 0u);

  // The query returns no status, so it leaves the message of an earlier
  // failure in place for the caller to read.
  uint64_t id = 0;
  EXPECT_EQ(M_DriverDeviceId(nullptr, &id), M_DRIVER_INVALID_ARGUMENT);
  const std::string before = lastError();
  ASSERT_FALSE(before.empty());
  EXPECT_EQ(M_DriverCApiVersion(), version);
  EXPECT_EQ(lastError(), before);
}

//===----------------------------------------------------------------------===//
// Errors
//===----------------------------------------------------------------------===//

TEST(DriverCApi, ErrorMessageIsEmptyAfterSuccessAndSizedAfterFailure) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  // Success clears the message.
  EXPECT_EQ(M_DriverLastErrorMessage(nullptr, 0), 0u);
  EXPECT_EQ(M_DriverLastErrorMessage(nullptr, 0), 0u);

  // A failure reports a message, and the required size includes the NUL.
  const M_DriverStatus status = M_DriverDeviceId(nullptr, nullptr);
  EXPECT_EQ(status, M_DRIVER_INVALID_ARGUMENT);
  const size_t needed = M_DriverLastErrorMessage(nullptr, 0);
  ASSERT_GT(needed, 1u);

  // A buffer too small is left alone, but still reports the required size.
  std::vector<char> small(1, '\x7f');
  EXPECT_EQ(M_DriverLastErrorMessage(small.data(), small.size()), needed);
  EXPECT_EQ(small[0], '\x7f');

  std::vector<char> message(needed, '\0');
  EXPECT_EQ(M_DriverLastErrorMessage(message.data(), message.size()), needed);
  EXPECT_STREQ(message.data(), lastError().c_str());
}

TEST(DriverCApi, RejectsNullAndOutOfRangeArguments) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  EXPECT_EQ(M_DriverDeviceId(device.get(), nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverDeviceSynchronize(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamCreate(device.get(), nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamSynchronize(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverEventCreate(nullptr, false, nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverBufferAllocate(device.get(), 0, false, nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverMemoryRegister(device.get(), nullptr, 0, nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverCopyHostToDevice(nullptr, nullptr, nullptr, 0),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverMappingDestroySync(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(
      M_DriverDeviceAllocationRange(device.get(), nullptr, nullptr, nullptr),
      M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamFromContextHandle(nullptr, nullptr),
            M_DRIVER_INVALID_ARGUMENT);

  // Every destroy refuses a null handle rather than treating it as a no op,
  // so a caller that lost track of a handle hears about it.
  EXPECT_EQ(M_DriverDeviceDestroy(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverStreamDestroy(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverEventDestroy(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverBufferDestroy(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverMappingDestroy(nullptr), M_DRIVER_INVALID_ARGUMENT);

  // An ordinal nobody has is an invalid argument, not a runtime error a
  // caller might retry, whether it is negative or one past the count.
  DeviceHandle bogus;
  EXPECT_EQ(M_DriverAcceleratorDeviceCreate(-1, bogus.receive()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverAcceleratorDeviceCreate(M_DriverAcceleratorCount(),
                                            bogus.receive()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());

  // There is one CPU device, named by 0 or by -1, and nothing else names it.
  DeviceHandle cpu;
  EXPECT_EQ(M_DriverCPUDeviceCreate(7, cpu.receive()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverCPUDeviceCreate(-2, cpu.receive()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCPUDeviceCreate(0, cpu.receive()), M_DRIVER_OK)
      << lastError();
}

//===----------------------------------------------------------------------===//
// Devices
//===----------------------------------------------------------------------===//

TEST(DriverCApi, CpuDeviceReportsIdentity) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  uint64_t id = 1234;
  ASSERT_EQ(M_DriverDeviceId(device.get(), &id), M_DRIVER_OK);
  EXPECT_EQ(id, 0u);

  const size_t expected = std::strlen("cpu") + 1;
  char api[16] = {};
  EXPECT_EQ(M_DriverDeviceApi(device.get(), api, sizeof(api)), expected);
  EXPECT_STREQ(api, "cpu");

  // The call that sized the name also cleared the error state, so a second
  // sizing call reports the same size.
  EXPECT_EQ(M_DriverDeviceApi(device.get(), nullptr, 0), expected);

  void *context_handle = nullptr;
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL context has no stable address to hand out.
  EXPECT_EQ(M_DriverDeviceContextHandle(device.get(), &context_handle),
            M_DRIVER_NOT_SUPPORTED);
#else
  EXPECT_EQ(M_DriverDeviceContextHandle(device.get(), &context_handle),
            M_DRIVER_OK);
  EXPECT_NE(context_handle, nullptr);
#endif
}

TEST(DriverCApi, PciBusIdIsEitherReportedOrExplained) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  // The CPU device has no PCI identity, so this must fail loudly rather than
  // report an empty string as a name.
  char busId[64] = {};
  size_t needed = M_DriverDevicePciBusId(device.get(), busId, sizeof(busId));
  EXPECT_EQ(needed, 0u);
  EXPECT_TRUE(busId[0] == '\0');
  EXPECT_FALSE(lastError().empty());

  EXPECT_EQ(M_DriverDevicePciBusId(nullptr, nullptr, 0), 0u);

  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle accelerator;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, accelerator.receive()),
            M_DRIVER_OK);
  needed = M_DriverDevicePciBusId(accelerator.get(), busId, sizeof(busId));
  if (needed == 0) {
    // A backend without NVML cannot answer; that must come with a reason.
    EXPECT_FALSE(lastError().empty());
    GTEST_SKIP() << "this backend cannot report a PCI bus id: " << lastError();
  }
  EXPECT_GT(needed, 1u);
  EXPECT_EQ(std::strlen(busId) + 1, needed);
}

TEST(DriverCApi, CpuDeviceHasNoUuid) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  // The host has no vendor identity, so the query refuses with a reason
  // rather than handing back bytes a caller could mistake for one.
  std::array<uint8_t, 16> uuid{};
  EXPECT_EQ(M_DriverDeviceUuid(device.get(), uuid.data()),
            M_DRIVER_NOT_SUPPORTED);
  EXPECT_FALSE(lastError().empty());

  EXPECT_EQ(M_DriverDeviceUuid(nullptr, uuid.data()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverDeviceUuid(device.get(), nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
}

TEST(DriverCApi, AcceleratorUuidIsStableAndDistinct) {
  const int64_t count = M_DriverAcceleratorCount();
  if (count == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle first;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, first.receive()), M_DRIVER_OK)
      << lastError();
  std::array<uint8_t, 16> uuid{};
  const M_DriverStatus status = M_DriverDeviceUuid(first.get(), uuid.data());
  if (status == M_DRIVER_NOT_SUPPORTED) {
    // A vendor driver that predates the query cannot answer, but it must
    // still say why.
    const std::string reason = lastError();
    EXPECT_FALSE(reason.empty()) << "refusing the query must explain why";
    GTEST_SKIP() << "this backend cannot report a device UUID: " << reason;
  }
  ASSERT_EQ(status, M_DRIVER_OK) << lastError();
  EXPECT_NE(uuid, (std::array<uint8_t, 16>{}))
      << "a reported UUID is never all zero";

  // The UUID names the device rather than the handle that asked, so a second
  // handle to the same ordinal reads the same bytes.
  DeviceHandle again;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, again.receive()), M_DRIVER_OK)
      << lastError();
  std::array<uint8_t, 16> reread{};
  ASSERT_EQ(M_DriverDeviceUuid(again.get(), reread.data()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(reread, uuid);

  if (count < 2)
    GTEST_SKIP() << "telling two devices apart needs two accelerators";

  DeviceHandle second;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(1, second.receive()), M_DRIVER_OK)
      << lastError();
  std::array<uint8_t, 16> other{};
  ASSERT_EQ(M_DriverDeviceUuid(second.get(), other.data()), M_DRIVER_OK)
      << lastError();
  EXPECT_NE(other, uuid) << "two devices never share a UUID";
}

TEST(DriverCApi, AcceleratorDeviceEnumeration) {
  const int64_t count = M_DriverAcceleratorCount();
  ASSERT_GE(count, 0);
  if (count == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();

  char api[32] = {};
  ASSERT_GT(M_DriverDeviceApi(device.get(), api, sizeof(api)), 1u);
  EXPECT_TRUE(std::string(api) == "cuda" || std::string(api) == "hip" ||
              std::string(api) == "metal")
      << "unexpected accelerator api: " << api;

  uint64_t id = 1234;
  ASSERT_EQ(M_DriverDeviceId(device.get(), &id), M_DRIVER_OK);
  EXPECT_EQ(id, 0u);

  char arch[64] = {};
  EXPECT_GT(M_DriverDeviceArchitectureName(device.get(), arch, sizeof(arch)),
            1u);
  EXPECT_GT(M_DriverDeviceModelName(device.get(), arch, sizeof(arch)), 1u);

  uint64_t total = 0;
  uint64_t free = 0;
  EXPECT_EQ(M_DriverDeviceMemoryInfo(device.get(), &total, &free), M_DRIVER_OK);
  EXPECT_GT(total, 0u);

  EXPECT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_OK);
}

TEST(DriverCApi, AcceleratorPeerQueriesOnOneDevice) {
  const int64_t count = M_DriverAcceleratorCount();
  if (count == 0)
    GTEST_SKIP() << "no accelerator on this host";
  if (count > 1)
    GTEST_SKIP() << "the pairwise cases run in the two device target";

  // A single device still answers the query, and enabling is idempotent.
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);
  bool canAccess = true;
  const M_DriverStatus status =
      M_DriverDeviceCanAccess(device.get(), device.get(), &canAccess);
  EXPECT_EQ(status, M_DRIVER_OK) << lastError();
  EXPECT_EQ(M_DriverEnableAllPeerAccess(), M_DRIVER_OK) << lastError();

  bool enabled = false;
  EXPECT_EQ(M_DriverAllPeerAccessEnabled(&enabled), M_DRIVER_OK);
}

//===----------------------------------------------------------------------===//
// Streams, buffers and copies
//===----------------------------------------------------------------------===//

/// Copies a pattern out to a buffer and back, over the device, and checks it
/// arrives unchanged. This is the round trip a host side cache performs: host
/// slab in, device memory, host slab out.
///
/// The transfer target is device memory, and the host side is a buffer the
/// driver allocated for DMA, because that is what the C ABI documents for the
/// directional copies. A pinned host allocation is the *host* endpoint and
/// never the destination of a host-to-device copy: the directional APIs want a
/// device-resident target, and only a backend whose host and device memory are
/// one pool would accept the other.
///
/// The copies run on `queue` when the caller passes one, and otherwise on a
/// stream this interface creates.
void expectRoundTrip(M_DriverDevice *device, M_DriverStream *queue = nullptr) {
  constexpr size_t kBytes = 4096;
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    expected[i] = static_cast<uint8_t>(i * 31 + 7);

  // The host endpoint of both copies, written through its host address.
  BufferHandle host;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/true,
                                   host.receive()),
            M_DRIVER_OK)
      << lastError();
  void *hostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(host.get(), &hostAddress), M_DRIVER_OK);
  ASSERT_NE(hostAddress, nullptr);
  uint64_t hostSize = 0;
  ASSERT_EQ(M_DriverBufferByteSize(host.get(), &hostSize), M_DRIVER_OK);
  EXPECT_EQ(hostSize, kBytes);
  bool hostIsHost = false;
  ASSERT_EQ(M_DriverBufferIsHost(host.get(), &hostIsHost), M_DRIVER_OK);
  EXPECT_TRUE(hostIsHost);
  auto *hostBytes = static_cast<uint8_t *>(hostAddress);
  std::memcpy(hostBytes, expected.data(), kBytes);

  BufferHandle buffer;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/false,
                                   buffer.receive()),
            M_DRIVER_OK)
      << lastError();

  void *address = nullptr;
  uint64_t size = 0;
  ASSERT_EQ(M_DriverBufferPointer(buffer.get(), &address), M_DRIVER_OK);
  ASSERT_EQ(M_DriverBufferByteSize(buffer.get(), &size), M_DRIVER_OK);
  EXPECT_EQ(size, kBytes);
  ASSERT_NE(address, nullptr);
  bool isHost = true;
  ASSERT_EQ(M_DriverBufferIsHost(buffer.get(), &isHost), M_DRIVER_OK);
  EXPECT_FALSE(isHost);

  StreamHandle created;
  if (queue == nullptr) {
    ASSERT_EQ(M_DriverStreamCreate(device, created.receive()), M_DRIVER_OK)
        << lastError();
    queue = created.get();
  }

  ASSERT_EQ(M_DriverCopyHostToDevice(queue, address, hostBytes, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(queue), M_DRIVER_OK) << lastError();

  // Overwrite the host buffer so the return leg has to do the work.
  std::memset(hostBytes, 0, kBytes);

  ASSERT_EQ(M_DriverCopyDeviceToHost(queue, hostBytes, address, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(queue), M_DRIVER_OK) << lastError();

  EXPECT_EQ(std::memcmp(hostBytes, expected.data(), kBytes), 0);
}

/// A pinned host allocation is host memory the device can DMA to and from: the
/// properties a caller branches on when choosing it as a staging endpoint.
void expectPinnedHostAllocation(M_DriverDevice *device) {
  constexpr uint64_t kBytes = 8192;
  BufferHandle pinned;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/true,
                                   pinned.receive()),
            M_DRIVER_OK)
      << lastError();

  bool isHost = false;
  ASSERT_EQ(M_DriverBufferIsHost(pinned.get(), &isHost), M_DRIVER_OK);
  EXPECT_TRUE(isHost);

  uint64_t size = 0;
  ASSERT_EQ(M_DriverBufferByteSize(pinned.get(), &size), M_DRIVER_OK);
  EXPECT_EQ(size, kBytes);

  void *address = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(pinned.get(), &address), M_DRIVER_OK);
  EXPECT_NE(address, nullptr);
}

TEST(DriverCApi, CpuRoundTripsAPatternThroughDeviceMemory) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  expectRoundTrip(device.get());
}

TEST(DriverCApi, CpuPinnedHostAllocationIsHostMemory) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  expectPinnedHostAllocation(device.get());
}

TEST(DriverCApi, AcceleratorRoundTripsAPatternThroughDeviceMemory) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);
  expectRoundTrip(device.get());
}

TEST(DriverCApi, AcceleratorPinnedHostAllocationIsHostMemory) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);
  expectPinnedHostAllocation(device.get());
}

TEST(DriverCApi, BorrowedBufferWrapsCallerMemoryWithoutOwningIt) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  std::vector<uint8_t> scratch(1024);
  for (size_t i = 0; i < scratch.size(); ++i)
    scratch[i] = static_cast<uint8_t>(i);

  BufferHandle borrowed;
  ASSERT_EQ(M_DriverBufferFromPointer(device.get(), scratch.data(),
                                      scratch.size(), /*is_host=*/true,
                                      borrowed.receive()),
            M_DRIVER_OK)
      << lastError();

  void *address = nullptr;
  uint64_t size = 0;
  ASSERT_EQ(M_DriverBufferPointer(borrowed.get(), &address), M_DRIVER_OK);
  ASSERT_EQ(M_DriverBufferByteSize(borrowed.get(), &size), M_DRIVER_OK);
  EXPECT_EQ(size, scratch.size());
  // Host memory the caller owns is wrapped, not copied.
  EXPECT_EQ(address, scratch.data());

  // A copy into the wrapped range lands in the caller's memory. The CPU
  // device's memory space is host memory, so a pageable range needs no
  // registration here; against an accelerator it would.
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK);
  std::vector<uint8_t> source(scratch.size(), 0xAB);
  ASSERT_EQ(M_DriverCopyHostToDevice(stream.get(), scratch.data(),
                                     source.data(), source.size()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);
  EXPECT_EQ(scratch, source);
}

TEST(DriverCApi, BorrowedBufferNeverFreesTheCallersMemory) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  std::vector<uint8_t> scratch(256, 0x5A);
  {
    BufferHandle borrowed;
    ASSERT_EQ(M_DriverBufferFromPointer(device.get(), scratch.data(),
                                        scratch.size(), /*is_host=*/true,
                                        borrowed.receive()),
              M_DRIVER_OK);
  }
  // Destroying the handle must leave the caller's memory untouched; a debug
  // allocator would report a bad free here.
  for (uint8_t byte : scratch)
    EXPECT_EQ(byte, 0x5A);
}

TEST(DriverCApi, StreamsReportTheirDeviceAndContext) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();

  DeviceHandle fromStream;
  ASSERT_EQ(M_DriverStreamDevice(stream.get(), fromStream.receive()),
            M_DRIVER_OK);

  char api[16] = {};
  ASSERT_GT(M_DriverDeviceApi(fromStream.get(), api, sizeof(api)), 1u);
  EXPECT_STREQ(api, "cpu");

#if !MODULAR_HAL_DEVICE_CONTEXT
  // The device a stream reports names the process-wide context, the one the
  // device handle holds, and not the stream's own, so synchronizing it drains
  // the device rather than the stream. The HAL build hands out no context
  // address to compare.
  void *deviceContext = nullptr;
  void *fromStreamContext = nullptr;
  ASSERT_EQ(M_DriverDeviceContextHandle(device.get(), &deviceContext),
            M_DRIVER_OK);
  ASSERT_EQ(M_DriverDeviceContextHandle(fromStream.get(), &fromStreamContext),
            M_DRIVER_OK);
  EXPECT_EQ(fromStreamContext, deviceContext);
#endif

  // A fresh stream is ordered behind the device's own work, and a device
  // handle can be waited on directly.
  EXPECT_EQ(M_DriverStreamWaitForDevice(stream.get(), device.get()),
            M_DRIVER_OK)
      << lastError();

  // Waiting for the stream's own context is a no-op, through the same handle
  // or through a second handle that adopted it. (A regression test as much as
  // a contract: the CPU backend's `enqueueWaitFor(ctx)` deadlocks on a
  // self-wait, so the C API has to be the one that no-ops.)
  EXPECT_EQ(M_DriverStreamWaitFor(stream.get(), stream.get()), M_DRIVER_OK)
      << lastError();

#if !MODULAR_HAL_DEVICE_CONTEXT
  void *context_handle = nullptr;
  ASSERT_EQ(M_DriverStreamDeviceContextHandle(stream.get(), &context_handle),
            M_DRIVER_OK);
  ASSERT_NE(context_handle, nullptr);
  EXPECT_NE(context_handle, deviceContext)
      << "a created stream has a context of its own";

  StreamHandle adopted;
  ASSERT_EQ(M_DriverStreamFromContextHandle(context_handle, adopted.receive()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamWaitFor(adopted.get(), stream.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamWaitForDevice(adopted.get(), device.get()),
            M_DRIVER_OK)
      << lastError();
#endif

  // The CPU backend has no native queue handle, so null is the answer, not an
  // error: a caller must be able to ask on every backend.
  void *native = nullptr;
  EXPECT_EQ(M_DriverStreamNativeHandle(stream.get(), &native), M_DRIVER_OK);
  EXPECT_EQ(native, nullptr);
}

TEST(DriverCApi, MixedBackendHandlesAreRefused) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle cpu;
  ASSERT_TRUE(makeCpuDevice(cpu));
  DeviceHandle accelerator;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, accelerator.receive()),
            M_DRIVER_OK)
      << lastError();
  StreamHandle cpuStream;
  ASSERT_EQ(M_DriverStreamCreate(cpu.get(), cpuStream.receive()), M_DRIVER_OK)
      << lastError();
  StreamHandle acceleratorStream;
  ASSERT_EQ(
      M_DriverStreamCreate(accelerator.get(), acceleratorStream.receive()),
      M_DRIVER_OK)
      << lastError();

  // Every backend casts the objects it is handed to its own types, so a mix
  // has to come back as a status here rather than be misread there.
  EXPECT_EQ(M_DriverStreamWaitFor(cpuStream.get(), acceleratorStream.get()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverStreamWaitFor(acceleratorStream.get(), cpuStream.get()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamWaitForDevice(acceleratorStream.get(), cpu.get()),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamWaitForDevice(cpuStream.get(), accelerator.get()),
            M_DRIVER_INVALID_ARGUMENT);

  bool canAccess = true;
  EXPECT_EQ(M_DriverDeviceCanAccess(accelerator.get(), cpu.get(), &canAccess),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverDeviceEnablePeerAccess(cpu.get(), accelerator.get()),
            M_DRIVER_INVALID_ARGUMENT);

  M_DriverEvent *acceleratorEvent = nullptr;
  ASSERT_EQ(M_DriverEventCreate(accelerator.get(), /*enable_timing=*/false,
                                &acceleratorEvent),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamRecordEvent(cpuStream.get(), acceleratorEvent),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamWaitForEvent(cpuStream.get(), acceleratorEvent),
            M_DRIVER_INVALID_ARGUMENT);
  M_DriverEvent *cpuEvent = nullptr;
  if (M_DriverEventCreate(cpu.get(), /*enable_timing=*/false, &cpuEvent) ==
      M_DRIVER_OK) {
    EXPECT_EQ(M_DriverStreamRecordEvent(acceleratorStream.get(), cpuEvent),
              M_DRIVER_INVALID_ARGUMENT);
    EXPECT_EQ(M_DriverStreamWaitForEvent(acceleratorStream.get(), cpuEvent),
              M_DRIVER_INVALID_ARGUMENT);
    double elapsed = -1.0;
    EXPECT_EQ(M_DriverEventElapsedTimeMs(cpuEvent, acceleratorEvent, &elapsed),
              M_DRIVER_INVALID_ARGUMENT);
    EXPECT_EQ(elapsed, -1.0);
    M_DriverEventDestroy(cpuEvent);
  }
  M_DriverEventDestroy(acceleratorEvent);

  // A peer copy whose source device is the CPU is a mix too, whichever side
  // the stream runs on.
  constexpr uint64_t kBytes = 64;
  std::vector<uint8_t> hostBytes(kBytes);
  BufferHandle buffer;
  ASSERT_EQ(M_DriverBufferAllocate(accelerator.get(), kBytes,
                                   /*pinned_host=*/false, buffer.receive()),
            M_DRIVER_OK)
      << lastError();
  void *address = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(buffer.get(), &address), M_DRIVER_OK);
  EXPECT_EQ(M_DriverCopyPeerToPeer(acceleratorStream.get(), accelerator.get(),
                                   address, cpu.get(), hostBytes.data(),
                                   kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
}

TEST(DriverCApi, StreamCreateValidatesItsArguments) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  // CPU has no legacy default stream to isolate from, so it hands out its
  // usual queue, and that queue works.
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  EXPECT_NE(stream.get(), nullptr);
  EXPECT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();

  // A null device is refused, and the out handle is cleared so a caller never
  // mistakes a stale value for a stream.
  M_DriverStream *noDevice = reinterpret_cast<M_DriverStream *>(0x1);
  EXPECT_EQ(M_DriverStreamCreate(nullptr, &noDevice),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(noDevice, nullptr);
  EXPECT_FALSE(lastError().empty());
}

TEST(DriverCApi, FreshStreamsAreNotCapturing) {
  bool capturing = true;
  EXPECT_EQ(M_DriverStreamIsCapturing(nullptr, &capturing),
            M_DRIVER_INVALID_ARGUMENT);

  DeviceHandle cpu;
  ASSERT_TRUE(makeCpuDevice(cpu));
  StreamHandle cpuStream;
  ASSERT_EQ(M_DriverStreamCreate(cpu.get(), cpuStream.receive()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamIsCapturing(cpuStream.get(), nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  ASSERT_EQ(M_DriverStreamIsCapturing(cpuStream.get(), &capturing), M_DRIVER_OK)
      << lastError();
  EXPECT_FALSE(capturing);

  if (M_DriverAcceleratorCount() == 0)
    return;
  DeviceHandle accelerator;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, accelerator.receive()),
            M_DRIVER_OK)
      << lastError();
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(accelerator.get(), stream.receive()),
            M_DRIVER_OK)
      << lastError();
  capturing = true;
  ASSERT_EQ(M_DriverStreamIsCapturing(stream.get(), &capturing), M_DRIVER_OK)
      << lastError();
  EXPECT_FALSE(capturing);

  // No context of ours runs an adopted native queue, so nothing can answer.
  void *native = nullptr;
  ASSERT_EQ(M_DriverStreamNativeHandle(stream.get(), &native), M_DRIVER_OK);
  StreamHandle adopted;
  if (native == nullptr ||
      M_DriverStreamFromNativeHandle(accelerator.get(), native,
                                     adopted.receive()) != M_DRIVER_OK)
    return;
  EXPECT_EQ(M_DriverStreamIsCapturing(adopted.get(), &capturing),
            M_DRIVER_NOT_SUPPORTED);
  EXPECT_FALSE(capturing);
  EXPECT_FALSE(lastError().empty());
}

TEST(DriverCApi, AcceleratorStreamsAreNonBlocking) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  expectRoundTrip(device.get(), stream.get());

  // CU_STREAM_NON_BLOCKING and hipStreamNonBlocking are both 0x1.
  constexpr unsigned kVendorNonBlocking = 0x1;
  char api[32] = {};
  ASSERT_GT(M_DriverDeviceApi(device.get(), api, sizeof(api)), 1u);
  if (std::strcmp(api, "cuda") != 0 && std::strcmp(api, "hip") != 0)
    return; // No vendor flags to query (Metal).
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim keeps its base queue";
#else
  // Every stream this interface creates is a pooled non blocking queue, so a
  // second one reads the same way and the two are different queues.
  StreamHandle another;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), another.receive()), M_DRIVER_OK)
      << lastError();
  for (M_DriverStream *created : {stream.get(), another.get()}) {
    const std::optional<unsigned> flags =
        vendorStreamFlags(device.get(), created);
    ASSERT_TRUE(flags.has_value())
        << "could not query the vendor queue's flags";
    EXPECT_EQ(*flags & kVendorNonBlocking, kVendorNonBlocking)
        << "the vendor queue must be created non blocking";
  }
  void *first = nullptr;
  void *second = nullptr;
  ASSERT_EQ(M_DriverStreamNativeHandle(stream.get(), &first), M_DRIVER_OK);
  ASSERT_EQ(M_DriverStreamNativeHandle(another.get(), &second), M_DRIVER_OK);
  EXPECT_NE(first, second);
#endif
}

TEST(DriverCApi, ThreadBindingIsExplicitAndReversible) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  ASSERT_EQ(M_DriverDeviceBindToCurrentThread(device.get()), M_DRIVER_OK)
      << lastError();
  // Binding the device that is already bound changes nothing.
  ASSERT_EQ(M_DriverDeviceBindToCurrentThread(device.get()), M_DRIVER_OK)
      << lastError();
  expectRoundTrip(device.get());

  M_DriverUnbindFromCurrentThread();
  // Nothing is bound now, so clearing the binding again is harmless and copies
  // keep working without one.
  M_DriverUnbindFromCurrentThread();
  expectRoundTrip(device.get());

  EXPECT_EQ(M_DriverDeviceBindToCurrentThread(nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
}

TEST(DriverCApi, AcceleratorThreadBindingKeepsCopiesWorking) {
  const int64_t count = M_DriverAcceleratorCount();
  if (count == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle first;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, first.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverDeviceBindToCurrentThread(first.get()), M_DRIVER_OK)
      << lastError();
  expectRoundTrip(first.get());

  // Declared here so the handle outlives the binding it may carry.
  DeviceHandle second;
  if (count >= 2) {
    // Binding another device replaces the binding, and copies on the first
    // device still work because every entry point makes its own context
    // current around the operation.
    ASSERT_EQ(M_DriverAcceleratorDeviceCreate(1, second.receive()), M_DRIVER_OK)
        << lastError();
    ASSERT_EQ(M_DriverDeviceBindToCurrentThread(second.get()), M_DRIVER_OK)
        << lastError();
    expectRoundTrip(first.get());
  }

  M_DriverUnbindFromCurrentThread();
}

TEST(DriverCApi, UnorderedPeerCopyValidatesItsArguments) {
  DeviceHandle dst;
  DeviceHandle src;
  ASSERT_TRUE(makeCpuDevice(dst));
  ASSERT_TRUE(makeCpuDevice(src));
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(dst.get(), stream.receive()), M_DRIVER_OK)
      << lastError();

  // Every call below is refused before the driver touches either range, so
  // host memory stands in for both.
  constexpr uint64_t kBytes = 64;
  std::vector<uint8_t> dstBytes(kBytes);
  std::vector<uint8_t> srcBytes(kBytes);

  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(nullptr, dst.get(), dstBytes.data(),
                                            src.get(), srcBytes.data(), kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(stream.get(), nullptr,
                                            dstBytes.data(), src.get(),
                                            srcBytes.data(), kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(stream.get(), dst.get(),
                                            dstBytes.data(), nullptr,
                                            srcBytes.data(), kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(stream.get(), dst.get(), nullptr,
                                            src.get(), srcBytes.data(), kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(stream.get(), dst.get(),
                                            dstBytes.data(), src.get(), nullptr,
                                            kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(stream.get(), dst.get(),
                                            dstBytes.data(), src.get(),
                                            srcBytes.data(), 0),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());

  // The CPU backend adopts any non null handle as a queue of its own, which
  // reaches the adopted stream path on a host with no vendor queue to hand
  // over. A backend that adopts nothing, such as the HAL shim, has no adopted
  // stream to refuse.
  StreamHandle adopted;
  if (M_DriverStreamFromNativeHandle(dst.get(),
                                     reinterpret_cast<void *>(0x1234),
                                     adopted.receive()) == M_DRIVER_OK) {
    EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(adopted.get(), dst.get(),
                                              dstBytes.data(), src.get(),
                                              srcBytes.data(), kBytes),
              M_DRIVER_NOT_SUPPORTED);
    EXPECT_FALSE(lastError().empty());
  }
}

/// Copies four pieces of different sizes, laid out back to back, out to device
/// memory in one batch and back in another, and checks every byte arrives
/// unchanged. Both batches run on `stream`, which runs on `device`.
void expectBatchRoundTrip(M_DriverDevice *device, M_DriverStream *stream) {
  constexpr std::array<uint64_t, 4> kSizes = {512, 1024, 256, 2304};
  constexpr uint64_t kBytes = 4096;
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    expected[i] = static_cast<uint8_t>(i * 31 + 7);

  BufferHandle host;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/true,
                                   host.receive()),
            M_DRIVER_OK)
      << lastError();
  void *hostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(host.get(), &hostAddress), M_DRIVER_OK);
  auto *hostBytes = static_cast<uint8_t *>(hostAddress);
  std::memcpy(hostBytes, expected.data(), kBytes);

  BufferHandle target;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/false,
                                   target.receive()),
            M_DRIVER_OK)
      << lastError();
  void *targetAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(target.get(), &targetAddress), M_DRIVER_OK);

  std::array<void *, 4> hostPieces{};
  std::array<const void *, 4> hostSources{};
  std::array<void *, 4> devicePieces{};
  std::array<const void *, 4> deviceSources{};
  uint64_t offset = 0;
  for (size_t i = 0; i < kSizes.size(); ++i) {
    hostPieces[i] = hostBytes + offset;
    hostSources[i] = hostBytes + offset;
    devicePieces[i] = static_cast<uint8_t *>(targetAddress) + offset;
    deviceSources[i] = static_cast<uint8_t *>(targetAddress) + offset;
    offset += kSizes[i];
  }
  ASSERT_EQ(offset, kBytes);

  ASSERT_EQ(M_DriverCopyBatch(stream, M_DRIVER_COPY_HOST_TO_DEVICE,
                              devicePieces.data(), hostSources.data(),
                              kSizes.data(), kSizes.size()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream), M_DRIVER_OK) << lastError();

  // Zero the host side so the return batch has to do the work.
  std::memset(hostBytes, 0, kBytes);

  ASSERT_EQ(M_DriverCopyBatch(stream, M_DRIVER_COPY_DEVICE_TO_HOST,
                              hostPieces.data(), deviceSources.data(),
                              kSizes.data(), kSizes.size()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream), M_DRIVER_OK) << lastError();

  EXPECT_EQ(std::memcmp(hostBytes, expected.data(), kBytes), 0);
}

TEST(DriverCApi, CpuBatchCopiesSeveralPiecesEachWay) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();

  expectBatchRoundTrip(device.get(), stream.get());

  // An empty batch enqueues nothing, so it needs no arrays.
  EXPECT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_HOST_TO_DEVICE,
                              nullptr, nullptr, nullptr, 0),
            M_DRIVER_OK)
      << lastError();
}

TEST(DriverCApi, BatchCopyValidatesItsArguments) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();

  // Every call below is refused before anything is enqueued, so host memory
  // stands in for both sides, and the destination staying zero shows that no
  // piece ran.
  constexpr uint64_t kPiece = 64;
  std::vector<uint8_t> dstBytes(2 * kPiece, 0);
  std::vector<uint8_t> srcBytes(2 * kPiece, 0x5a);
  std::array<void *, 2> dsts = {dstBytes.data(), dstBytes.data() + kPiece};
  std::array<const void *, 2> srcs = {srcBytes.data(),
                                      srcBytes.data() + kPiece};
  std::array<uint64_t, 2> sizes = {kPiece, kPiece};

  EXPECT_EQ(M_DriverCopyBatch(nullptr, M_DRIVER_COPY_HOST_TO_DEVICE,
                              dsts.data(), srcs.data(), sizes.data(), 2),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_HOST_TO_DEVICE,
                              nullptr, nullptr, nullptr, 1),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());

  // A bad piece is named by its index, and the good piece before it is not
  // enqueued either.
  std::array<void *, 2> nullAddress = {dstBytes.data(), nullptr};
  EXPECT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_HOST_TO_DEVICE,
                              nullAddress.data(), srcs.data(), sizes.data(), 2),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_NE(lastError().find("piece 1"), std::string::npos) << lastError();
  std::array<uint64_t, 2> zeroSize = {kPiece, 0};
  EXPECT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_HOST_TO_DEVICE,
                              dsts.data(), srcs.data(), zeroSize.data(), 2),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_NE(lastError().find("piece 1"), std::string::npos) << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(dstBytes, std::vector<uint8_t>(2 * kPiece, 0));

  // The CPU backend adopts any non null handle as a queue of its own, which
  // reaches the adopted stream path on a host with no vendor queue to hand
  // over. A backend that adopts nothing, such as the HAL shim, has no adopted
  // stream to refuse.
  StreamHandle adopted;
  if (M_DriverStreamFromNativeHandle(device.get(),
                                     reinterpret_cast<void *>(0x1234),
                                     adopted.receive()) == M_DRIVER_OK) {
    EXPECT_EQ(M_DriverCopyBatch(adopted.get(), M_DRIVER_COPY_HOST_TO_DEVICE,
                                dsts.data(), srcs.data(), sizes.data(), 2),
              M_DRIVER_NOT_SUPPORTED);
    EXPECT_FALSE(lastError().empty());
  }
}

TEST(DriverCApi, AcceleratorBatchCopiesSeveralPiecesEachWay) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  expectBatchRoundTrip(device.get(), stream.get());
}

//===----------------------------------------------------------------------===//
// Events
//===----------------------------------------------------------------------===//

TEST(DriverCApi, AdoptedNativeStreamAcceptsOrderingOperations) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);

  StreamHandle driverStream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), driverStream.receive()),
            M_DRIVER_OK);

  void *native = nullptr;
  ASSERT_EQ(M_DriverStreamNativeHandle(driverStream.get(), &native),
            M_DRIVER_OK);
  if (native == nullptr)
    GTEST_SKIP() << "this backend exposes no native queue handle";

  StreamHandle adopted;
  const M_DriverStatus adoptedStatus =
      M_DriverStreamFromNativeHandle(device.get(), native, adopted.receive());
  if (adoptedStatus != M_DRIVER_OK) {
    // Not every backend can adopt a queue it does not own (Metal cannot); that
    // must come with a reason rather than a handle that orders nothing.
    EXPECT_EQ(adopted.get(), nullptr);
    EXPECT_FALSE(lastError().empty());
    GTEST_SKIP() << "this backend does not adopt foreign queues: "
                 << lastError();
  }

  // A native handle stream names its device like any other stream, but it
  // has no context of its own to hand out, because the one created beside the
  // queue is not ordered against it.
  DeviceHandle adoptedDevice;
  ASSERT_EQ(M_DriverStreamDevice(adopted.get(), adoptedDevice.receive()),
            M_DRIVER_OK)
      << lastError();
  void *deviceContext = nullptr;
  void *adoptedDeviceContext = nullptr;
  ASSERT_EQ(M_DriverDeviceContextHandle(device.get(), &deviceContext),
            M_DRIVER_OK);
  ASSERT_EQ(
      M_DriverDeviceContextHandle(adoptedDevice.get(), &adoptedDeviceContext),
      M_DRIVER_OK);
  EXPECT_EQ(adoptedDeviceContext, deviceContext);
  void *adoptedContext = nullptr;
  EXPECT_EQ(M_DriverStreamDeviceContextHandle(adopted.get(), &adoptedContext),
            M_DRIVER_NOT_SUPPORTED);
  EXPECT_FALSE(lastError().empty());

  // What this checks is that every ordering entry point accepts a stream that
  // wraps a borrowed queue, in both directions, and that the copies such a
  // stream cannot serve are refused. It cannot check *where* the wait landed: a
  // wait silently enqueued on the context this interface created beside the
  // native queue would also return success, which is why the routing (an event
  // recorded on the source queue, waited on at the destination) is worth
  // re-observing on a CUDA or HIP host, where the queue can be inspected after
  // the fact.
  EXPECT_EQ(M_DriverStreamWaitFor(adopted.get(), driverStream.get()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamWaitFor(driverStream.get(), adopted.get()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamWaitFor(adopted.get(), adopted.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverStreamWaitForDevice(adopted.get(), device.get()),
            M_DRIVER_OK)
      << lastError();

  // Event waits and records work on the borrowed queue too.
  M_DriverEvent *event = nullptr;
  if (M_DriverEventCreate(device.get(), /*enable_timing=*/false, &event) ==
      M_DRIVER_OK) {
    EXPECT_EQ(M_DriverStreamRecordEvent(adopted.get(), event), M_DRIVER_OK)
        << lastError();
    EXPECT_EQ(M_DriverStreamWaitForEvent(adopted.get(), event), M_DRIVER_OK)
        << lastError();
    M_DriverEventDestroy(event);
  }
  // So does recording a new event, which lands on the borrowed queue like an
  // existing one.
  M_DriverEvent *recorded = nullptr;
  EXPECT_EQ(M_DriverStreamRecordNewEvent(adopted.get(), &recorded), M_DRIVER_OK)
      << lastError();
  if (recorded != nullptr) {
    EXPECT_EQ(M_DriverEventSynchronize(recorded), M_DRIVER_OK) << lastError();
    M_DriverEventDestroy(recorded);
  }

  // Copies do not: they must be enqueued on the context that owns the
  // destination's memory, which is why a native handle is the wrong tool for
  // them.
  constexpr uint64_t kBytes = 4096;
  BufferHandle buffer;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), kBytes, /*pinned_host=*/false,
                                   buffer.receive()),
            M_DRIVER_OK);
  void *address = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(buffer.get(), &address), M_DRIVER_OK);
  EXPECT_EQ(M_DriverCopyDeviceToDevice(adopted.get(), address, address, kBytes),
            M_DRIVER_NOT_SUPPORTED);
}

TEST(DriverCApi, CpuEventsRecordAndMeasure) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK);
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);

  M_DriverEvent *raw = nullptr;
  const M_DriverStatus status =
      M_DriverEventCreate(device.get(), /*enable_timing=*/true, &raw);
  if (status != M_DRIVER_OK) {
    // The CPU backend is allowed to have no event model; what it must not do
    // is lie about a handle it did not create.
    EXPECT_EQ(raw, nullptr);
    EXPECT_FALSE(lastError().empty());
    GTEST_SKIP() << "this backend has no events: " << lastError();
  }
  EXPECT_EQ(M_DriverStreamRecordEvent(stream.get(), raw), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverEventSynchronize(raw), M_DRIVER_OK) << lastError();

  bool ready = false;
  EXPECT_EQ(M_DriverEventIsReady(raw, &ready), M_DRIVER_OK) << lastError();
  EXPECT_TRUE(ready);

  // Timed events measure a duration between two fired events. The backends
  // without timestamping (the CPU device among them) must fail loudly instead
  // of reporting a bogus duration.
  M_DriverEvent *end = nullptr;
  if (M_DriverEventCreate(device.get(), /*enable_timing=*/true, &end) ==
          M_DRIVER_OK &&
      M_DriverStreamRecordEvent(stream.get(), end) == M_DRIVER_OK &&
      M_DriverEventSynchronize(end) == M_DRIVER_OK) {
    double elapsed = -1.0;
    const M_DriverStatus elapsedStatus =
        M_DriverEventElapsedTimeMs(raw, end, &elapsed);
    if (elapsedStatus == M_DRIVER_OK) {
      EXPECT_GE(elapsed, 0.0);
    } else {
      EXPECT_EQ(elapsed, -1.0) << "a failed measurement must not write a "
                                  "duration";
      EXPECT_FALSE(lastError().empty());
    }
    M_DriverEventDestroy(end);
  }

  M_DriverEventDestroy(raw);
}

TEST(DriverCApi, AcceleratorTimedEventsMeasureADuration) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);

  char api[32] = {};
  ASSERT_GT(M_DriverDeviceApi(device.get(), api, sizeof(api)), 1u);

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK);

  bool ready = false;
  M_DriverEvent *start = nullptr;
  const M_DriverStatus timingStatus =
      M_DriverEventCreate(device.get(), /*enable_timing=*/true, &start);
  if (timingStatus != M_DRIVER_OK) {
    // A backend that cannot timestamp (Metal today) must refuse at creation,
    // with a reason, rather than hand back an event that silently never
    // measures.
    EXPECT_EQ(start, nullptr);
    const std::string reason = lastError();
    EXPECT_FALSE(reason.empty()) << "refusing timing must explain why";

    // Ordering-only events still have to work on that backend: they are what
    // a copy stream orders against.
    ASSERT_EQ(
        M_DriverEventCreate(device.get(), /*enable_timing=*/false, &start),
        M_DRIVER_OK)
        << lastError();
    ASSERT_EQ(M_DriverStreamRecordEvent(stream.get(), start), M_DRIVER_OK)
        << lastError();
    EXPECT_EQ(M_DriverEventSynchronize(start), M_DRIVER_OK) << lastError();
    EXPECT_EQ(M_DriverEventIsReady(start, &ready), M_DRIVER_OK) << lastError();
    EXPECT_TRUE(ready) << "a synchronized event must report itself ready";
    M_DriverEventDestroy(start);
    GTEST_SKIP() << "this backend cannot timestamp events: " << reason;
  }

  M_DriverEvent *end = nullptr;
  ASSERT_EQ(M_DriverEventCreate(device.get(), /*enable_timing=*/true, &end),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamRecordEvent(stream.get(), start), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamRecordEvent(stream.get(), end), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverEventSynchronize(start), M_DRIVER_OK) << lastError();
  ASSERT_EQ(M_DriverEventSynchronize(end), M_DRIVER_OK) << lastError();

  EXPECT_EQ(M_DriverEventIsReady(end, &ready), M_DRIVER_OK) << lastError();
  EXPECT_TRUE(ready);

  double elapsed = -1.0;
#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL build accepts timing at creation and fails the measurement, which
  // the header allows.
  EXPECT_NE(M_DriverEventElapsedTimeMs(start, end, &elapsed), M_DRIVER_OK);
  EXPECT_FALSE(lastError().empty());
  M_DriverEventDestroy(start);
  M_DriverEventDestroy(end);
  GTEST_SKIP() << "the HAL device context cannot time events";
#else
  EXPECT_EQ(M_DriverEventElapsedTimeMs(start, end, &elapsed), M_DRIVER_OK)
      << lastError();
  EXPECT_GE(elapsed, 0.0);

  M_DriverEventDestroy(start);
  M_DriverEventDestroy(end);
#endif
}

//===----------------------------------------------------------------------===//
// Completion flags
//===----------------------------------------------------------------------===//

TEST(DriverCApi, CpuRefusesCompletionFlags) {
  DeviceHandle device;
  ASSERT_EQ(M_DriverCPUDeviceCreate(0, device.receive()), M_DRIVER_OK);
  M_DriverCompletionFlag *flag = nullptr;
  EXPECT_NE(M_DriverCompletionFlagCreate(device.get(), &flag), M_DRIVER_OK);
  EXPECT_EQ(flag, nullptr);
}

TEST(DriverCApi, AcceleratorStreamWaitsForFlag) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK);
  M_DriverCompletionFlag *flag = nullptr;
  const M_DriverStatus created =
      M_DriverCompletionFlagCreate(device.get(), &flag);
  if (MODULAR_HAL_DEVICE_CONTEXT && created == M_DRIVER_NOT_SUPPORTED)
    GTEST_SKIP() << "no HAL completion flags on this backend: " << lastError();
  ASSERT_EQ(created, M_DRIVER_OK) << lastError();

  // A flag waited on from another backend's stream is refused.
  DeviceHandle cpu;
  ASSERT_EQ(M_DriverCPUDeviceCreate(0, cpu.receive()), M_DRIVER_OK);
  StreamHandle cpuStream;
  ASSERT_EQ(M_DriverStreamCreate(cpu.get(), cpuStream.receive()), M_DRIVER_OK);
  EXPECT_EQ(M_DriverStreamWaitFlag(cpuStream.get(), flag, 1),
            M_DRIVER_INVALID_ARGUMENT);

  // Already past the target.
  ASSERT_EQ(M_DriverCompletionFlagSignal(flag, 5), M_DRIVER_OK);
  ASSERT_EQ(M_DriverStreamWaitFlag(stream.get(), flag, 3), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();

  // Gated until the host signals.
  ASSERT_EQ(M_DriverStreamWaitFlag(stream.get(), flag, 9), M_DRIVER_OK)
      << lastError();
  M_DriverEvent *done = nullptr;
  ASSERT_EQ(M_DriverStreamRecordNewEvent(stream.get(), &done), M_DRIVER_OK)
      << lastError();
  bool ready = true;
  ASSERT_EQ(M_DriverEventIsReady(done, &ready), M_DRIVER_OK) << lastError();
  EXPECT_FALSE(ready) << "the stream ran past an unsatisfied wait";
  ASSERT_EQ(M_DriverCompletionFlagSignal(flag, 9), M_DRIVER_OK);
  EXPECT_EQ(M_DriverEventSynchronize(done), M_DRIVER_OK) << lastError();

  M_DriverEventDestroy(done);
  EXPECT_EQ(M_DriverCompletionFlagDestroy(flag), M_DRIVER_OK);
}

//===----------------------------------------------------------------------===//
// Host memory registration
//===----------------------------------------------------------------------===//

TEST(DriverCApi, RegisteringCallerMemoryMakesItDeviceAddressable) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim registers no host memory";
#endif
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "memory registration needs an accelerator";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);

  // Page-aligned, page-granular: the range a MAP_SHARED slab would have.
  constexpr size_t kBytes = 1 << 16;
  void *slab = nullptr;
  ASSERT_EQ(posix_memalign(&slab, 1 << 16, kBytes), 0);
  std::memset(slab, 0x11, kBytes);

  MappingHandle mapping;
  ASSERT_EQ(
      M_DriverMemoryRegister(device.get(), slab, kBytes, mapping.receive()),
      M_DRIVER_OK)
      << lastError();

  void *address = nullptr;
  uint64_t size = 0;
  EXPECT_EQ(M_DriverMappingPointer(mapping.get(), &address), M_DRIVER_OK);
  EXPECT_EQ(M_DriverMappingByteSize(mapping.get(), &size), M_DRIVER_OK);
  EXPECT_EQ(size, kBytes);
  EXPECT_NE(address, nullptr);

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK);

  // The registered range must be usable as a copy source and destination on
  // the device, through the device visible address, and the bytes must land
  // in the caller's pages. The slab goes up with its pattern, is cleared, and
  // comes back, so a copy that silently did nothing would be caught.
  BufferHandle buffer;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), kBytes, /*pinned_host=*/false,
                                   buffer.receive()),
            M_DRIVER_OK)
      << lastError();
  void *deviceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(buffer.get(), &deviceAddress), M_DRIVER_OK);

  ASSERT_EQ(
      M_DriverCopyHostToDevice(stream.get(), deviceAddress, address, kBytes),
      M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  std::memset(slab, 0, kBytes);

  ASSERT_EQ(
      M_DriverCopyDeviceToHost(stream.get(), address, deviceAddress, kBytes),
      M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  std::vector<uint8_t> expected(kBytes, 0x11);
  EXPECT_EQ(std::memcmp(slab, expected.data(), kBytes), 0)
      << "the device to host copy into the registered range must reach the "
         "caller's pages";

  // Unregistering must leave the caller's pages alone and free them to be
  // reused; the region stays writable from the host.
  mapping.receive();
  std::memset(slab, 0x22, kBytes);

  // The unregister is queued; the device synchronize runs it. A caller that
  // reuses an address relies on this sequence.
  ASSERT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_OK)
      << lastError();
  M_DriverMapping *again = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &again),
            M_DRIVER_OK)
      << "the same range must register again after the device synchronized: "
      << lastError();

  // The synchronous unregister needs no device synchronize.
  ASSERT_EQ(M_DriverMappingDestroySync(again), M_DRIVER_OK) << lastError();
  M_DriverMapping *onceMore = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &onceMore),
            M_DRIVER_OK)
      << "the same range must register again right after a synchronous "
         "unregister: "
      << lastError();
  ASSERT_EQ(M_DriverMappingDestroySync(onceMore), M_DRIVER_OK) << lastError();

  // A queued unregister runs inside the next synchronous one on the device.
  M_DriverMapping *queued = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &queued),
            M_DRIVER_OK)
      << lastError();
  M_DriverMappingDestroy(queued);
  void *other = nullptr;
  ASSERT_EQ(posix_memalign(&other, 1 << 16, kBytes), 0);
  M_DriverMapping *otherMapping = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), other, kBytes, &otherMapping),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverMappingDestroySync(otherMapping), M_DRIVER_OK)
      << lastError();
  M_DriverMapping *afterQueued = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &afterQueued),
            M_DRIVER_OK)
      << "the queued unregister must have run inside the synchronous one: "
      << lastError();
  ASSERT_EQ(M_DriverMappingDestroySync(afterQueued), M_DRIVER_OK)
      << lastError();
  free(other);
  free(slab);
}

namespace M::Driver {

// In the driver's namespace for its types.
TEST(DriverCApi, SyncUnregisterLeavesAPendingErrorInPlace) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim registers no host memory";
#else
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "memory registration needs an accelerator";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  // The same process-wide context the handle holds, reached the way the
  // driver's own callers reach it and not through the opaque handle, which
  // the header says never to dereference.
  ErrorOr<DeviceContextRef> shared =
      DeviceContext::getOrCreate(Device::gpuAPI, 0);
  ASSERT_FALSE(shared.isError()) << shared.getError();
  DeviceContextRef context = shared.takeValue();

  constexpr size_t kBytes = 1 << 16;
  void *slab = nullptr;
  ASSERT_EQ(posix_memalign(&slab, 1 << 16, kBytes), 0);

  // The unregister must neither trip on the shared context's error nor
  // consume it.
  M_DriverMapping *underError = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &underError),
            M_DRIVER_OK)
      << lastError();
  ErrorOrSuccess planted = plantPendingError(*context);
  ASSERT_FALSE(planted.isError()) << planted.getError();
  ASSERT_EQ(M_DriverMappingDestroySync(underError), M_DRIVER_OK) << lastError();
  // The error is still pending, so the next call that checks for one, here
  // the synchronize, reports it, and only then does the context run clean.
  EXPECT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_RUNTIME_ERROR)
      << "the pending error must still be there for the next synchronous "
         "call to report";
  EXPECT_FALSE(lastError().empty());
  // Registering goes after the report; its success shows the unregister ran
  // under the error.
  M_DriverMapping *afterError = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &afterError),
            M_DRIVER_OK)
      << "the range must register again, so it was unregistered while the "
         "error was pending: "
      << lastError();
  ASSERT_EQ(M_DriverMappingDestroySync(afterError), M_DRIVER_OK) << lastError();
  free(slab);
#endif
}

TEST(DriverCApi, EventCreateLeavesAPendingErrorInPlace) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim has no shared context to record into";
#else
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "event creation under a pending error needs an accelerator";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  ErrorOr<DeviceContextRef> shared =
      DeviceContext::getOrCreate(Device::gpuAPI, 0);
  ASSERT_FALSE(shared.isError()) << shared.getError();
  DeviceContextRef context = shared.takeValue();

  // An error another user of the shared context recorded must neither fail
  // event creation nor be consumed by it, and the event must still record
  // and complete on a stream of the device.
  ErrorOrSuccess planted = plantPendingError(*context);
  ASSERT_FALSE(planted.isError()) << planted.getError();
  M_DriverEvent *event = nullptr;
  // Untimed: Metal events cannot time.
  ASSERT_EQ(M_DriverEventCreate(device.get(), /*enable_timing=*/false, &event),
            M_DRIVER_OK)
      << "event creation must not take the shared context's pending error: "
      << lastError();
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamRecordEvent(stream.get(), event), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverEventSynchronize(event), M_DRIVER_OK) << lastError();
  M_DriverEventDestroy(event);

  // The error is still pending, so the next call that checks the shared
  // context reports it, and only then does the context run clean.
  EXPECT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_RUNTIME_ERROR)
      << "the pending error must still be there for the next synchronous "
         "call to report";
  EXPECT_FALSE(lastError().empty());
  ASSERT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_OK)
      << lastError();
#endif
}

// Capture is not in the C interface, so the test begins it through the
// driver on a context it then adopts as a stream.
TEST(DriverCApi, AdoptedStreamReportsCapture) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL device context has no stream capture";
#else
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "stream capture needs an accelerator";

  ErrorOr<DeviceContextRef> created = DeviceContext::create(Device::gpuAPI, 0);
  ASSERT_FALSE(created.isError()) << created.getError();
  DeviceContextRef context = created.takeValue();
  if (context->device().api() != Device::cudaAPI &&
      context->device().api() != Device::hipAPI)
    GTEST_SKIP() << "stream capture is implemented for CUDA and HIP only";
  ErrorOr<DeviceStreamRef> base = context->stream();
  ASSERT_FALSE(base.isError()) << base.getError();
  DeviceStreamRef driverStream = base.takeValue();

  StreamHandle adopted;
  ASSERT_EQ(
      M_DriverStreamFromContextHandle(context.getPointer(), adopted.receive()),
      M_DRIVER_OK)
      << lastError();

  bool capturing = true;
  ASSERT_EQ(M_DriverStreamIsCapturing(adopted.get(), &capturing), M_DRIVER_OK)
      << lastError();
  EXPECT_FALSE(capturing);

  ErrorOrSuccess begun = driverStream->beginCapture();
  ASSERT_FALSE(begun.isError()) << begun.getError();
  ASSERT_EQ(M_DriverStreamIsCapturing(adopted.get(), &capturing), M_DRIVER_OK)
      << lastError();
  EXPECT_TRUE(capturing);

  ErrorOr<DeviceGraphRef> graph = driverStream->endCapture();
  ASSERT_FALSE(graph.isError()) << graph.getError();
  ASSERT_EQ(M_DriverStreamIsCapturing(adopted.get(), &capturing), M_DRIVER_OK)
      << lastError();
  EXPECT_FALSE(capturing);
#endif
}

TEST(DriverCApi, AdoptedRecordingContextReportsCapture) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL device context has no graph builder";
#else
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "graph building needs an accelerator";

  ErrorOr<DeviceContextRef> created = DeviceContext::create(Device::gpuAPI, 0);
  ASSERT_FALSE(created.isError()) << created.getError();
  DeviceContextRef context = created.takeValue();
  if (context->device().api() != Device::cudaAPI &&
      context->device().api() != Device::hipAPI)
    GTEST_SKIP() << "graph building is implemented for CUDA and HIP only";
  ErrorOr<DeviceGraphBuilderRef> builder = context->createGraphBuilder();
  ASSERT_FALSE(builder.isError()) << builder.getError();
  ErrorOr<DeviceGraphNode> root = (*builder)->addEmpty(/*dependencies=*/{});
  ASSERT_FALSE(root.isError()) << root.getError();
  ErrorOr<DeviceContextRef> recording = (*builder)->recordingContext(*root);
  ASSERT_FALSE(recording.isError()) << recording.getError();

  StreamHandle adopted;
  ASSERT_EQ(M_DriverStreamFromContextHandle((*recording).getPointer(),
                                            adopted.receive()),
            M_DRIVER_OK)
      << lastError();
  bool capturing = false;
  ASSERT_EQ(M_DriverStreamIsCapturing(adopted.get(), &capturing), M_DRIVER_OK)
      << lastError();
  EXPECT_TRUE(capturing);
#endif
}

// `mo.wait_host_value` rebuilds the driver's flag object from the handle, so
// the test reads it back as one rather than through the C interface.
TEST(DriverCApi, FlagGraphHandleIsTheDriverFlag) {
  uint64_t handle = 1;
  EXPECT_EQ(M_DriverCompletionFlagGraphHandle(nullptr, &handle),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(handle, 0u);

  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);
  M_DriverCompletionFlag *flag = nullptr;
  const M_DriverStatus created =
      M_DriverCompletionFlagCreate(device.get(), &flag);
  if (MODULAR_HAL_DEVICE_CONTEXT && created == M_DRIVER_NOT_SUPPORTED)
    GTEST_SKIP() << "no HAL completion flags on this backend: " << lastError();
  ASSERT_EQ(created, M_DRIVER_OK) << lastError();

  EXPECT_EQ(M_DriverCompletionFlagGraphHandle(flag, nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  ASSERT_EQ(M_DriverCompletionFlagGraphHandle(flag, &handle), M_DRIVER_OK)
      << lastError();
  ASSERT_NE(handle, 0u);
  auto *driverFlag = reinterpret_cast<CompletionFlag *>(handle);

  ASSERT_EQ(M_DriverCompletionFlagSignal(flag, 7), M_DRIVER_OK);
  EXPECT_EQ(driverFlag->load(), 7u);

  ErrorOr<DeviceContextRef> shared =
      DeviceContext::getOrCreate(Device::gpuAPI, 0);
  ASSERT_FALSE(shared.isError()) << shared.getError();
  ErrorOr<DeviceStreamRef> stream = (*shared)->stream();
  ASSERT_FALSE(stream.isError()) << stream.getError();
  ErrorOrSuccess waited = (*stream)->enqueueWaitOnFlag(*driverFlag, 7);
  ASSERT_FALSE(waited.isError()) << waited.getError();
  ErrorOrSuccess synced = (*stream)->synchronize();
  ASSERT_FALSE(synced.isError()) << synced.getError();

  EXPECT_EQ(M_DriverCompletionFlagDestroy(flag), M_DRIVER_OK);
}

} // namespace M::Driver

TEST(DriverCApi, BatchCopyThroughRegisteredMemoryLeavesTheRegistrationAlone) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim registers no host memory";
#endif
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "memory registration needs an accelerator";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK);

  constexpr size_t kBytes = 1 << 16;
  void *slab = nullptr;
  ASSERT_EQ(posix_memalign(&slab, 1 << 16, kBytes), 0);
  auto *slabBytes = static_cast<uint8_t *>(slab);
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    expected[i] = static_cast<uint8_t>(i * 13 + 5);
  std::memcpy(slab, expected.data(), kBytes);

  M_DriverMapping *mapping = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &mapping),
            M_DRIVER_OK)
      << lastError();

  // A separate stream: the batch runs on a context other than the one that
  // registered the range.
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  BufferHandle buffer;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), kBytes, /*pinned_host=*/false,
                                   buffer.receive()),
            M_DRIVER_OK)
      << lastError();
  void *deviceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(buffer.get(), &deviceAddress), M_DRIVER_OK);
  auto *deviceBytes = static_cast<uint8_t *>(deviceAddress);

  // Repeated, so a wrapper that freed or unregistered the range would break a
  // later round.
  constexpr std::array<uint64_t, 3> kSizes = {4096, 8192, kBytes - 12288};
  std::array<void *, 3> hostDsts{};
  std::array<const void *, 3> hostSrcs{};
  std::array<void *, 3> deviceDsts{};
  std::array<const void *, 3> deviceSrcs{};
  uint64_t offset = 0;
  for (size_t i = 0; i < kSizes.size(); ++i) {
    hostDsts[i] = slabBytes + offset;
    hostSrcs[i] = slabBytes + offset;
    deviceDsts[i] = deviceBytes + offset;
    deviceSrcs[i] = deviceBytes + offset;
    offset += kSizes[i];
  }
  for (int round = 0; round < 3; ++round) {
    ASSERT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_HOST_TO_DEVICE,
                                deviceDsts.data(), hostSrcs.data(),
                                kSizes.data(), kSizes.size()),
              M_DRIVER_OK)
        << lastError();
    ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
        << lastError();
    std::memset(slab, 0, kBytes);
    ASSERT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_DEVICE_TO_HOST,
                                hostDsts.data(), deviceSrcs.data(),
                                kSizes.data(), kSizes.size()),
              M_DRIVER_OK)
        << lastError();
    ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
        << lastError();
    ASSERT_EQ(std::memcmp(slab, expected.data(), kBytes), 0) << round;
  }

  // The mapping still owns the one registration: unregister, then re-register,
  // must both succeed.
  M_DriverMappingDestroy(mapping);
  ASSERT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  M_DriverMapping *again = nullptr;
  ASSERT_EQ(M_DriverMemoryRegister(device.get(), slab, kBytes, &again),
            M_DRIVER_OK)
      << lastError();
  M_DriverMappingDestroy(again);
  ASSERT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_OK)
      << lastError();
  free(slab);
}

//===----------------------------------------------------------------------===//
// The device's own stream
//===----------------------------------------------------------------------===//

using M::ErrorOr;
using M::Driver::DeviceContext;
using M::Driver::DeviceContextRef;

namespace {

/// Writes the native queue handle of `stream`, failing the test when the call
/// fails.
void *nativeHandleOf(const M_DriverStream *stream) {
  void *native = nullptr;
  EXPECT_EQ(M_DriverStreamNativeHandle(stream, &native), M_DRIVER_OK)
      << lastError();
  return native;
}

} // namespace

TEST(DriverCApi, DeviceStreamValidatesItsArguments) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  EXPECT_EQ(M_DriverDeviceStream(device.get(), nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  M_DriverStream *stale = reinterpret_cast<M_DriverStream *>(0x1);
  EXPECT_EQ(M_DriverDeviceStream(nullptr, &stale), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(stale, nullptr);
  EXPECT_FALSE(lastError().empty());
  stale = reinterpret_cast<M_DriverStream *>(0x1);
  EXPECT_EQ(M_DriverDeviceStream(device.get(), &stale), M_DRIVER_NOT_SUPPORTED);
  EXPECT_EQ(stale, nullptr);
}

TEST(DriverCApi, DeviceStreamIsMaxsSharedQueue) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  StreamHandle shared;
  ASSERT_EQ(M_DriverDeviceStream(device.get(), shared.receive()), M_DRIVER_OK)
      << lastError();

  // The stream reports the device it was retained from.
  DeviceHandle fromStream;
  ASSERT_EQ(M_DriverStreamDevice(shared.get(), fromStream.receive()),
            M_DRIVER_OK);
  uint64_t id = 99;
  ASSERT_EQ(M_DriverDeviceId(fromStream.get(), &id), M_DRIVER_OK);
  EXPECT_EQ(id, 0u);

#if MODULAR_HAL_DEVICE_CONTEXT
  // The HAL build hands out no context address, so the classic adoption this
  // compares against does not exist there; the ordering test covers it.
  GTEST_SKIP() << "the HAL build cannot adopt a context by address";
#else
  // It is the queue a caller would get by adopting the device's context,
  // which is MAX's: the same context and the same native queue.
  void *deviceContext = nullptr;
  ASSERT_EQ(M_DriverDeviceContextHandle(device.get(), &deviceContext),
            M_DRIVER_OK);
  void *streamContext = nullptr;
  ASSERT_EQ(M_DriverStreamDeviceContextHandle(shared.get(), &streamContext),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(streamContext, deviceContext);

  StreamHandle adopted;
  ASSERT_EQ(M_DriverStreamFromContextHandle(deviceContext, adopted.receive()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(nativeHandleOf(shared.get()), nativeHandleOf(adopted.get()));

  // A second retain is a second handle on the same queue.
  StreamHandle again;
  ASSERT_EQ(M_DriverDeviceStream(device.get(), again.receive()), M_DRIVER_OK)
      << lastError();
  EXPECT_NE(again.get(), shared.get());
  EXPECT_EQ(nativeHandleOf(again.get()), nativeHandleOf(shared.get()));
#endif
}

TEST(DriverCApi, DeviceStreamOrdersAfterSharedQueueWork) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  constexpr uint64_t kBytes = 64ull << 20;
  constexpr uint8_t kFill = 0xAB;

  // MAX's own context for the device: the fill is enqueued the way a model's
  // write is, through the C++ driver, with no C API call involved.
  ErrorOr<DeviceContextRef> max =
      DeviceContext::getOrCreate(M::Driver::Device::gpuAPI, 0);
  ASSERT_FALSE(max.isError()) << max.getError();
  ErrorOr<M::Driver::DeviceBufferRef> target =
      (*max)->enqueueCreateBuffer(kBytes);
  ASSERT_FALSE(target.isError()) << target.getError();

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  StreamHandle shared;
  ASSERT_EQ(M_DriverDeviceStream(device.get(), shared.receive()), M_DRIVER_OK)
      << lastError();
  BufferHandle landing;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), kBytes, /*pinned_host=*/true,
                                   landing.receive()),
            M_DRIVER_OK)
      << lastError();
  void *landingAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(landing.get(), &landingAddress), M_DRIVER_OK);
  std::memset(landingAddress, 0, kBytes);

  // Enough fills queued ahead of the copy that a copy on any other queue
  // would start while they run and read a stale value.
  for (int i = 0; i < 64; ++i)
    ASSERT_FALSE((*max)->enqueueSetMemory(**target, 0x11, 1).isError());
  ASSERT_FALSE((*max)->enqueueSetMemory(**target, kFill, 1).isError());
  // No host synchronization between the fill and the copy: only the queue
  // orders them.
  ASSERT_EQ(M_DriverCopyDeviceToHost(shared.get(), landingAddress,
                                     (*target)->ptr(), kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(shared.get()), M_DRIVER_OK)
      << lastError();

  const auto *bytes = static_cast<const uint8_t *>(landingAddress);
  size_t wrong = 0;
  for (uint64_t i = 0; i < kBytes; ++i)
    wrong += bytes[i] != kFill;
  EXPECT_EQ(wrong, 0u) << "the copy ran ahead of MAX's fill";
}

TEST(DriverCApi, StreamCreateIsNotDeviceStream) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  StreamHandle shared;
  ASSERT_EQ(M_DriverDeviceStream(device.get(), shared.receive()), M_DRIVER_OK)
      << lastError();
  StreamHandle created;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), created.receive()), M_DRIVER_OK)
      << lastError();

  void *sharedNative = nativeHandleOf(shared.get());
  void *createdNative = nativeHandleOf(created.get());
  if (sharedNative == nullptr && createdNative == nullptr)
    GTEST_SKIP() << "this backend exposes no native queue handle";
#if MODULAR_HAL_DEVICE_CONTEXT
  // A plugin that allows one context per device makes createContext share
  // the device's, and with it MAX's queue.
  if (sharedNative == createdNative)
    GTEST_SKIP() << "this HAL plugin shares one context per device, so a "
                    "created stream is MAX's queue";
#endif
  EXPECT_NE(sharedNative, createdNative);
}

//===----------------------------------------------------------------------===//
// Memset
//===----------------------------------------------------------------------===//

namespace {

/// Fills 4 KiB of device memory with a pattern through a copy, zeroes the
/// middle half with a memset, and checks only that half changed.
void expectMemsetFillsRangeOnly(M_DriverDevice *device) {
  constexpr uint64_t kBytes = 4096;
  constexpr uint64_t kBegin = 1024;
  constexpr uint64_t kEnd = 3072;

  BufferHandle host;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/true,
                                   host.receive()),
            M_DRIVER_OK)
      << lastError();
  void *hostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(host.get(), &hostAddress), M_DRIVER_OK);
  std::memset(hostAddress, 0xAB, kBytes);

  BufferHandle target;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/false,
                                   target.receive()),
            M_DRIVER_OK)
      << lastError();
  void *targetAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(target.get(), &targetAddress), M_DRIVER_OK);

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device, stream.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamWaitForDevice(stream.get(), device), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverCopyHostToDevice(stream.get(), targetAddress, hostAddress,
                                     kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverMemset(stream.get(),
                           static_cast<uint8_t *>(targetAddress) + kBegin, 0,
                           kEnd - kBegin),
            M_DRIVER_OK)
      << lastError();
  // The copy reads the host buffer when it runs, so it must finish before the
  // buffer is overwritten for the return leg.
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  std::memset(hostAddress, 0x11, kBytes);
  ASSERT_EQ(M_DriverCopyDeviceToHost(stream.get(), hostAddress, targetAddress,
                                     kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();

  const auto *bytes = static_cast<const uint8_t *>(hostAddress);
  for (uint64_t i = 0; i < kBytes; ++i) {
    const uint8_t expected = (i >= kBegin && i < kEnd) ? 0 : 0xAB;
    ASSERT_EQ(bytes[i], expected) << "at byte " << i;
  }
}

} // namespace

TEST(DriverCApi, CpuMemsetFillsRangeOnly) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  expectMemsetFillsRangeOnly(device.get());
}

TEST(DriverCApi, AcceleratorMemsetFillsRangeOnly) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  expectMemsetFillsRangeOnly(device.get());
}

TEST(DriverCApi, MemsetValidatesItsArguments) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  uint8_t bytes[16] = {};
  EXPECT_EQ(M_DriverMemset(nullptr, bytes, 1, sizeof bytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverMemset(stream.get(), nullptr, 1, sizeof bytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverMemset(stream.get(), bytes, 1, 0),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);
  for (uint8_t byte : bytes)
    EXPECT_EQ(byte, 0);

  // See BatchCopyValidatesItsArguments for why the CPU backend adopts this.
  StreamHandle adopted;
  if (M_DriverStreamFromNativeHandle(device.get(),
                                     reinterpret_cast<void *>(0x1234),
                                     adopted.receive()) == M_DRIVER_OK) {
    EXPECT_EQ(M_DriverMemset(adopted.get(), bytes, 1, sizeof bytes),
              M_DRIVER_NOT_SUPPORTED);
    EXPECT_FALSE(lastError().empty());
  }
}

//===----------------------------------------------------------------------===//
// Device-to-device batch copies
//===----------------------------------------------------------------------===//

namespace {

/// Copies three disjoint pieces of one device buffer into another, at
/// different offsets than they came from, with one batch.
void expectDeviceToDeviceBatch(M_DriverDevice *device) {
  constexpr uint64_t kBytes = 4096;
  std::vector<uint8_t> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    pattern[i] = static_cast<uint8_t>(i * 13 + 5);

  BufferHandle host;
  ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/true,
                                   host.receive()),
            M_DRIVER_OK)
      << lastError();
  void *hostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(host.get(), &hostAddress), M_DRIVER_OK);
  std::memcpy(hostAddress, pattern.data(), kBytes);

  BufferHandle source;
  BufferHandle destination;
  for (BufferHandle *buffer : {&source, &destination})
    ASSERT_EQ(M_DriverBufferAllocate(device, kBytes, /*pinned_host=*/false,
                                     buffer->receive()),
              M_DRIVER_OK)
        << lastError();
  void *src = nullptr;
  void *dst = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(source.get(), &src), M_DRIVER_OK);
  ASSERT_EQ(M_DriverBufferPointer(destination.get(), &dst), M_DRIVER_OK);

  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device, stream.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamWaitForDevice(stream.get(), device), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverCopyHostToDevice(stream.get(), src, hostAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  // Zero the destination so pieces the batch leaves alone read as 0.
  std::memset(hostAddress, 0, kBytes);
  ASSERT_EQ(M_DriverCopyHostToDevice(stream.get(), dst, hostAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  std::memset(hostAddress, 0x11, kBytes);

  struct Piece {
    uint64_t from, to, size;
  };
  const std::array<Piece, 3> pieces = {
      {{0, 2048, 512}, {1024, 0, 1000}, {3000, 3500, 96}}};
  std::array<void *, 3> dsts{};
  std::array<const void *, 3> srcs{};
  std::array<uint64_t, 3> sizes{};
  std::vector<uint8_t> expected(kBytes, 0);
  for (size_t i = 0; i < pieces.size(); ++i) {
    dsts[i] = static_cast<uint8_t *>(dst) + pieces[i].to;
    srcs[i] = static_cast<const uint8_t *>(src) + pieces[i].from;
    sizes[i] = pieces[i].size;
    std::memcpy(expected.data() + pieces[i].to, pattern.data() + pieces[i].from,
                pieces[i].size);
  }
  ASSERT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_DEVICE_TO_DEVICE,
                              dsts.data(), srcs.data(), sizes.data(),
                              pieces.size()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverCopyDeviceToHost(stream.get(), hostAddress, dst, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(std::memcmp(hostAddress, expected.data(), kBytes), 0);
}

} // namespace

TEST(DriverCApi, CpuBatchD2DSameDevice) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  expectDeviceToDeviceBatch(device.get());
}

TEST(DriverCApi, AcceleratorBatchD2DSameDevice) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  expectDeviceToDeviceBatch(device.get());
}

TEST(DriverCApi, BatchD2DUnknownDirectionRejected) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  uint8_t dstBytes[8] = {};
  const uint8_t srcBytes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  void *dsts[1] = {dstBytes};
  const void *srcs[1] = {srcBytes};
  const uint64_t sizes[1] = {sizeof dstBytes};
  EXPECT_EQ(M_DriverCopyBatch(stream.get(),
                              static_cast<M_DriverCopyDirection>(3), dsts, srcs,
                              sizes, 1),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());
  // Zero size pieces are still refused in the new direction.
  const uint64_t zero[1] = {0};
  EXPECT_EQ(M_DriverCopyBatch(stream.get(), M_DRIVER_COPY_DEVICE_TO_DEVICE,
                              dsts, srcs, zero, 1),
            M_DRIVER_INVALID_ARGUMENT);
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);
  for (uint8_t byte : dstBytes)
    EXPECT_EQ(byte, 0);
}

//===----------------------------------------------------------------------===//
// Device attributes and allocator memory
//===----------------------------------------------------------------------===//

TEST(DriverCApi, MultiprocessorCountMatchesDeviceContext) {
  DeviceHandle cpu;
  ASSERT_TRUE(makeCpuDevice(cpu));
  int64_t value = -1;
  EXPECT_EQ(
      M_DriverDeviceGetAttribute(
          cpu.get(), M_DRIVER_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, &value),
      M_DRIVER_NOT_SUPPORTED);
  EXPECT_FALSE(lastError().empty());
  EXPECT_EQ(
      M_DriverDeviceGetAttribute(
          nullptr, M_DRIVER_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, &value),
      M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(
      M_DriverDeviceGetAttribute(
          cpu.get(), M_DRIVER_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr),
      M_DRIVER_INVALID_ARGUMENT);

  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  // An attribute a newer header names is unsupported, not a crash.
  EXPECT_EQ(
      M_DriverDeviceGetAttribute(
          device.get(), static_cast<M_DriverDeviceAttribute>(1000), &value),
      M_DRIVER_NOT_SUPPORTED);

  ErrorOr<DeviceContextRef> max =
      DeviceContext::getOrCreate(M::Driver::Device::gpuAPI, 0);
  ASSERT_FALSE(max.isError()) << max.getError();
  ErrorOr<int> expected = (*max)->device().getAttribute(
      M::Driver::Device::Attribute::MULTIPROCESSOR_COUNT);
  const M_DriverStatus status = M_DriverDeviceGetAttribute(
      device.get(), M_DRIVER_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, &value);
  if (expected.isError()) {
    EXPECT_EQ(status, M_DRIVER_RUNTIME_ERROR);
    GTEST_SKIP() << "this backend does not report the multiprocessor count: "
                 << expected.getError();
  }
  ASSERT_EQ(status, M_DRIVER_OK) << lastError();
  EXPECT_EQ(value, static_cast<int64_t>(*expected));
  EXPECT_GT(value, 0);
}

namespace {

/// The allocator's figures equal the context's own query, which answers from
/// the memory manager when the device has one and from the driver otherwise.
void expectAllocatorMemoryInfoMatchesContext(M_DriverDevice *device,
                                             DeviceContext &context) {
  ErrorOr<M::Driver::Device::MemInfo> before = context.getMemoryInfo();
  ASSERT_FALSE(before.isError()) << before.getError();
  uint64_t budget = 0;
  uint64_t allocatable = 0;
  ASSERT_EQ(M_DriverDeviceAllocatorMemoryInfo(device, &budget, &allocatable),
            M_DRIVER_OK)
      << lastError();
  ErrorOr<M::Driver::Device::MemInfo> after = context.getMemoryInfo();
  ASSERT_FALSE(after.isError()) << after.getError();

  EXPECT_EQ(budget, static_cast<uint64_t>(before->total));
  if (!before->free.has_value()) {
    EXPECT_EQ(allocatable, M_DRIVER_MEMORY_UNKNOWN);
    return;
  }
  // The driver's free figure moves with every other process on the device,
  // so it is compared exactly only when it held still around the call. The
  // memory manager's figure only moves with this process's allocations.
  if (before->free == after->free)
    EXPECT_EQ(allocatable, static_cast<uint64_t>(*before->free));
  EXPECT_LE(allocatable, budget);

  // Either out-pointer may be null.
  EXPECT_EQ(M_DriverDeviceAllocatorMemoryInfo(device, nullptr, nullptr),
            M_DRIVER_OK)
      << lastError();
}

} // namespace

TEST(DriverCApi, CpuAllocatorMemoryInfoMatchesContext) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  ErrorOr<DeviceContextRef> context =
      DeviceContext::getOrCreate(M::Driver::Device::cpuAPI, 0);
  ASSERT_FALSE(context.isError()) << context.getError();
  expectAllocatorMemoryInfoMatchesContext(device.get(), **context);
  EXPECT_EQ(M_DriverDeviceAllocatorMemoryInfo(nullptr, nullptr, nullptr),
            M_DRIVER_INVALID_ARGUMENT);
}

TEST(DriverCApi, AcceleratorAllocatorMemoryInfoMatchesContext) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  ErrorOr<DeviceContextRef> context =
      DeviceContext::getOrCreate(M::Driver::Device::gpuAPI, 0);
  ASSERT_FALSE(context.isError()) << context.getError();
  expectAllocatorMemoryInfoMatchesContext(device.get(), **context);

  // Allocating through the interface draws on the same pool the figures
  // describe, so with a memory manager the allocatable bytes drop.
  uint64_t before = 0;
  ASSERT_EQ(M_DriverDeviceAllocatorMemoryInfo(device.get(), nullptr, &before),
            M_DRIVER_OK);
  BufferHandle held;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), 16ull << 20,
                                   /*pinned_host=*/false, held.receive()),
            M_DRIVER_OK)
      << lastError();
  uint64_t after = 0;
  ASSERT_EQ(M_DriverDeviceAllocatorMemoryInfo(device.get(), nullptr, &after),
            M_DRIVER_OK);
  if (before != M_DRIVER_MEMORY_UNKNOWN && after != M_DRIVER_MEMORY_UNKNOWN)
    EXPECT_LE(after, before);
}

//===----------------------------------------------------------------------===//
// Host functions
//===----------------------------------------------------------------------===//

namespace {

struct HostCopy {
  const void *from = nullptr;
  std::vector<uint8_t> into;
  std::atomic<int> calls{0};
};

void copyHostBytes(void *userData) {
  auto *copy = static_cast<HostCopy *>(userData);
  std::memcpy(copy->into.data(), copy->from, copy->into.size());
  copy->calls.fetch_add(1);
}

} // namespace

TEST(DriverCApi, HostFunctionValidatesItsArguments) {
  DeviceHandle device;
  ASSERT_TRUE(makeCpuDevice(device));
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();
  HostCopy copy;
  EXPECT_EQ(M_DriverStreamEnqueueHostFunction(nullptr, copyHostBytes, &copy),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverStreamEnqueueHostFunction(stream.get(), nullptr, &copy),
            M_DRIVER_INVALID_ARGUMENT);
  // The CPU device has no host functions, and a refused call never runs fn.
  EXPECT_EQ(
      M_DriverStreamEnqueueHostFunction(stream.get(), copyHostBytes, &copy),
      M_DRIVER_NOT_SUPPORTED);
  EXPECT_FALSE(lastError().empty());
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);
  EXPECT_EQ(copy.calls.load(), 0);
}

TEST(DriverCApi, HostFunctionRunsAfterPriorWork) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "no accelerator on this host";
  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device.get(), stream.receive()), M_DRIVER_OK)
      << lastError();

  // Large, so a function that ran ahead of the copy would see it unfinished.
  constexpr uint64_t kBytes = 32ull << 20;
  std::vector<uint8_t> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    pattern[i] = static_cast<uint8_t>(i * 7 + 3);

  BufferHandle host;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), kBytes, /*pinned_host=*/true,
                                   host.receive()),
            M_DRIVER_OK)
      << lastError();
  void *hostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(host.get(), &hostAddress), M_DRIVER_OK);
  std::memcpy(hostAddress, pattern.data(), kBytes);
  BufferHandle filled;
  ASSERT_EQ(M_DriverBufferAllocate(device.get(), kBytes, /*pinned_host=*/false,
                                   filled.receive()),
            M_DRIVER_OK)
      << lastError();
  void *filledAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(filled.get(), &filledAddress), M_DRIVER_OK);
  ASSERT_EQ(M_DriverStreamWaitForDevice(stream.get(), device.get()),
            M_DRIVER_OK);
  ASSERT_EQ(M_DriverCopyHostToDevice(stream.get(), filledAddress, hostAddress,
                                     kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);
  std::memset(hostAddress, 0, kBytes);

  HostCopy copy;
  copy.from = hostAddress;
  copy.into.assign(kBytes, 0);
  ASSERT_EQ(M_DriverCopyDeviceToHost(stream.get(), hostAddress, filledAddress,
                                     kBytes),
            M_DRIVER_OK)
      << lastError();
  const M_DriverStatus status =
      M_DriverStreamEnqueueHostFunction(stream.get(), copyHostBytes, &copy);
  char api[32] = {};
  ASSERT_GT(M_DriverDeviceApi(device.get(), api, sizeof(api)), 1u);
  if (std::strcmp(api, "cuda") != 0 && std::strcmp(api, "hip") != 0) {
    EXPECT_EQ(status, M_DRIVER_NOT_SUPPORTED);
    ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK);
    EXPECT_EQ(copy.calls.load(), 0);
    GTEST_SKIP() << "host functions run on CUDA and HIP only";
  }
  ASSERT_EQ(status, M_DRIVER_OK) << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(copy.calls.load(), 1);
  EXPECT_EQ(copy.into, pattern) << "the function ran ahead of the copy";
}
