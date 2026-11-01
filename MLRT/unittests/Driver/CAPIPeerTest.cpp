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
// Tests for the driver C API across two accelerators, which are the peer
// access queries and the peer copies. The target that builds this asks for
// two devices, so a host with fewer is skipped before anything runs, and a
// pair that cannot reach each other skips the cases that need it.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/CAPI/DriverCAPI.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "CAPITestHelpers.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace capi_test;

TEST(DriverCApiPeer, PeerAccessQueries) {
  if (M_DriverAcceleratorCount() < 2)
    GTEST_SKIP() << "peer access between two devices needs two accelerators";

  DeviceHandle first;
  DeviceHandle second;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, first.receive()), M_DRIVER_OK);
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(1, second.receive()), M_DRIVER_OK);

  bool canAccess = false;
  ASSERT_EQ(M_DriverDeviceCanAccess(first.get(), second.get(), &canAccess),
            M_DRIVER_OK)
      << lastError();
  // The driver refuses to enable a pair the hardware cannot connect, and
  // enabling every pair at once fails as a whole on such a host, so the rest
  // of this test needs the pair.
  if (!canAccess)
    GTEST_SKIP() << "devices cannot access each other's memory";

  // Per-pair enable, in both directions, is idempotent.
  EXPECT_EQ(M_DriverDeviceEnablePeerAccess(first.get(), second.get()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverDeviceEnablePeerAccess(first.get(), second.get()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverDeviceEnablePeerAccess(second.get(), first.get()),
            M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverDeviceEnablePeerAccess(first.get(), nullptr),
            M_DRIVER_INVALID_ARGUMENT);

  ASSERT_EQ(M_DriverEnableAllPeerAccess(), M_DRIVER_OK) << lastError();
  bool enabled = false;
  ASSERT_EQ(M_DriverAllPeerAccessEnabled(&enabled), M_DRIVER_OK);
  EXPECT_TRUE(enabled);
}

TEST(DriverCApiPeer, UnorderedPeerCopyOrderedByEvents) {
  if (M_DriverAcceleratorCount() < 2)
    GTEST_SKIP() << "a peer copy needs two accelerators";

  DeviceHandle first;
  DeviceHandle second;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, first.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(1, second.receive()), M_DRIVER_OK)
      << lastError();

  bool canAccess = false;
  ASSERT_EQ(M_DriverDeviceCanAccess(second.get(), first.get(), &canAccess),
            M_DRIVER_OK)
      << lastError();
  if (!canAccess)
    GTEST_SKIP() << "devices cannot access each other's memory";
  ASSERT_EQ(M_DriverDeviceEnablePeerAccess(second.get(), first.get()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverDeviceEnablePeerAccess(first.get(), second.get()),
            M_DRIVER_OK)
      << lastError();

  constexpr uint64_t kBytes = 4096;
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    expected[i] = static_cast<uint8_t>(i * 31 + 7);

  // The producer fills the source on the first device from a pinned host
  // buffer and marks the point where the source is ready.
  BufferHandle sourceHost;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kBytes, /*pinned_host=*/true,
                                   sourceHost.receive()),
            M_DRIVER_OK)
      << lastError();
  void *sourceHostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(sourceHost.get(), &sourceHostAddress),
            M_DRIVER_OK);
  std::memcpy(sourceHostAddress, expected.data(), kBytes);

  BufferHandle source;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kBytes, /*pinned_host=*/false,
                                   source.receive()),
            M_DRIVER_OK)
      << lastError();
  void *sourceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(source.get(), &sourceAddress), M_DRIVER_OK);

  StreamHandle producer;
  ASSERT_EQ(M_DriverStreamCreate(first.get(), producer.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverCopyHostToDevice(producer.get(), sourceAddress,
                                     sourceHostAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  M_DriverEvent *produced = nullptr;
  ASSERT_EQ(
      M_DriverEventCreate(first.get(), /*enable_timing=*/false, &produced),
      M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamRecordEvent(producer.get(), produced), M_DRIVER_OK)
      << lastError();

  // The copy stream on the second device waits for that point itself, because
  // the unordered copy leaves the ordering against the source to the caller.
  StreamHandle copier;
  ASSERT_EQ(M_DriverStreamCreate(second.get(), copier.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamWaitForEvent(copier.get(), produced), M_DRIVER_OK)
      << lastError();

  BufferHandle destination;
  ASSERT_EQ(M_DriverBufferAllocate(second.get(), kBytes, /*pinned_host=*/false,
                                   destination.receive()),
            M_DRIVER_OK)
      << lastError();
  void *destinationAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(destination.get(), &destinationAddress),
            M_DRIVER_OK);

  // A peer copy is enqueued on the destination's own context, so a stream of
  // the source device is refused.
  EXPECT_EQ(M_DriverCopyPeerToPeerUnordered(producer.get(), second.get(),
                                            destinationAddress, first.get(),
                                            sourceAddress, kBytes),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_FALSE(lastError().empty());

  ASSERT_EQ(M_DriverCopyPeerToPeerUnordered(copier.get(), second.get(),
                                            destinationAddress, first.get(),
                                            sourceAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  M_DriverEvent *copied = nullptr;
  ASSERT_EQ(M_DriverEventCreate(second.get(), /*enable_timing=*/false, &copied),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamRecordEvent(copier.get(), copied), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverEventSynchronize(copied), M_DRIVER_OK) << lastError();

  BufferHandle landing;
  ASSERT_EQ(M_DriverBufferAllocate(second.get(), kBytes, /*pinned_host=*/true,
                                   landing.receive()),
            M_DRIVER_OK)
      << lastError();
  void *landingAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(landing.get(), &landingAddress), M_DRIVER_OK);
  ASSERT_EQ(M_DriverCopyDeviceToHost(copier.get(), landingAddress,
                                     destinationAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(copier.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(std::memcmp(landingAddress, expected.data(), kBytes), 0);

  M_DriverEventDestroy(copied);
  M_DriverEventDestroy(produced);
}

/// A source produced by a queue of slow copies on the first device's shared
/// context, read by an ordered peer copy on the second device that waits for
/// nothing itself. The ordered variant has to wait for the source device's
/// shared context, which is the only thing that makes the bytes right.
TEST(DriverCApiPeer, OrderedPeerCopyWaitsForTheSourceContext) {
  if (M_DriverAcceleratorCount() < 2)
    GTEST_SKIP() << "a peer copy needs two accelerators";

  DeviceHandle first;
  DeviceHandle second;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, first.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(1, second.receive()), M_DRIVER_OK)
      << lastError();

  bool canAccess = false;
  ASSERT_EQ(M_DriverDeviceCanAccess(second.get(), first.get(), &canAccess),
            M_DRIVER_OK)
      << lastError();
  if (!canAccess)
    GTEST_SKIP() << "devices cannot access each other's memory";
  ASSERT_EQ(M_DriverDeviceEnablePeerAccess(second.get(), first.get()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverDeviceEnablePeerAccess(first.get(), second.get()),
            M_DRIVER_OK)
      << lastError();

  // The producer is the first device's shared context itself, adopted as a
  // stream, because that is the context the ordered copy promises to wait on.
  void *producerContext = nullptr;
  ASSERT_EQ(M_DriverDeviceContextHandle(first.get(), &producerContext),
            M_DRIVER_OK)
      << lastError();
  StreamHandle producer;
  if (M_DriverStreamFromContextHandle(producerContext, producer.receive()) !=
      M_DRIVER_OK)
    GTEST_SKIP() << "this build cannot adopt a context: " << lastError();

  constexpr uint64_t kBytes = 4096;
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    expected[i] = static_cast<uint8_t>(i * 29 + 3);

  // Slow work ahead of the source, a stack of large copies the producer has
  // to finish before the pattern lands, so a copy that skipped the wait would
  // read the zeroed source.
  constexpr uint64_t kSlowBytes = 64u << 20;
  BufferHandle slowHost;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kSlowBytes,
                                   /*pinned_host=*/true, slowHost.receive()),
            M_DRIVER_OK)
      << lastError();
  void *slowHostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(slowHost.get(), &slowHostAddress),
            M_DRIVER_OK);
  BufferHandle slowDevice;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kSlowBytes,
                                   /*pinned_host=*/false, slowDevice.receive()),
            M_DRIVER_OK)
      << lastError();
  void *slowDeviceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(slowDevice.get(), &slowDeviceAddress),
            M_DRIVER_OK);

  BufferHandle source;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kBytes, /*pinned_host=*/false,
                                   source.receive()),
            M_DRIVER_OK)
      << lastError();
  void *sourceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(source.get(), &sourceAddress), M_DRIVER_OK);
  BufferHandle sourceHost;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kBytes, /*pinned_host=*/true,
                                   sourceHost.receive()),
            M_DRIVER_OK)
      << lastError();
  void *sourceHostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(sourceHost.get(), &sourceHostAddress),
            M_DRIVER_OK);

  // Everything the copy needs on the second device exists before the slow
  // stack is queued, so only the peer copy call itself sits between the
  // producer's enqueue and the copy, and the stack cannot drain in between.
  StreamHandle copier;
  ASSERT_EQ(M_DriverStreamCreate(second.get(), copier.receive()), M_DRIVER_OK)
      << lastError();
  BufferHandle destination;
  ASSERT_EQ(M_DriverBufferAllocate(second.get(), kBytes, /*pinned_host=*/false,
                                   destination.receive()),
            M_DRIVER_OK)
      << lastError();
  void *destinationAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(destination.get(), &destinationAddress),
            M_DRIVER_OK);
  BufferHandle landing;
  ASSERT_EQ(M_DriverBufferAllocate(second.get(), kBytes, /*pinned_host=*/true,
                                   landing.receive()),
            M_DRIVER_OK)
      << lastError();
  void *landingAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(landing.get(), &landingAddress), M_DRIVER_OK);

  // The source starts zeroed on the device, then the slow stack, then the
  // pattern, all on the producer and none of it waited for by the host.
  std::vector<uint8_t> zeros(kBytes, 0);
  std::memcpy(sourceHostAddress, zeros.data(), kBytes);
  ASSERT_EQ(M_DriverCopyHostToDevice(producer.get(), sourceAddress,
                                     sourceHostAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(producer.get()), M_DRIVER_OK)
      << lastError();
  std::memcpy(sourceHostAddress, expected.data(), kBytes);
  for (int round = 0; round < 32; ++round) {
    ASSERT_EQ(M_DriverCopyHostToDevice(producer.get(), slowDeviceAddress,
                                       slowHostAddress, kSlowBytes),
              M_DRIVER_OK)
        << lastError();
  }
  ASSERT_EQ(M_DriverCopyHostToDevice(producer.get(), sourceAddress,
                                     sourceHostAddress, kBytes),
            M_DRIVER_OK)
      << lastError();

  ASSERT_EQ(M_DriverCopyPeerToPeer(copier.get(), second.get(),
                                   destinationAddress, first.get(),
                                   sourceAddress, kBytes),
            M_DRIVER_OK)
      << lastError();

  ASSERT_EQ(M_DriverCopyDeviceToHost(copier.get(), landingAddress,
                                     destinationAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(copier.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(std::memcmp(landingAddress, expected.data(), kBytes), 0)
      << "the ordered peer copy read the source before the producer wrote it";

  // The bracket's other half is that the producer waits for the copy, so its
  // queue drains only once the copy is done, and both sides end clean.
  ASSERT_EQ(M_DriverStreamSynchronize(producer.get()), M_DRIVER_OK)
      << lastError();
}

namespace M::Driver {

// In the driver's namespace for its types.
TEST(DriverCApiPeer, UnorderedPeerCopyLeavesAPendingErrorInPlace) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim has no shared context to record into";
#else
  if (M_DriverAcceleratorCount() < 2)
    GTEST_SKIP() << "a peer copy needs two accelerators";

  DeviceHandle first;
  DeviceHandle second;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, first.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(1, second.receive()), M_DRIVER_OK)
      << lastError();
  bool canAccess = false;
  ASSERT_EQ(M_DriverDeviceCanAccess(second.get(), first.get(), &canAccess),
            M_DRIVER_OK)
      << lastError();
  if (!canAccess)
    GTEST_SKIP() << "devices cannot access each other's memory";
  ASSERT_EQ(M_DriverDeviceEnablePeerAccess(second.get(), first.get()),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverDeviceEnablePeerAccess(first.get(), second.get()),
            M_DRIVER_OK)
      << lastError();
  ErrorOr<DeviceContextRef> shared =
      DeviceContext::getOrCreate(Device::gpuAPI, 0);
  ASSERT_FALSE(shared.isError()) << shared.getError();
  DeviceContextRef sourceContext = shared.takeValue();

  constexpr uint64_t kBytes = 4096;
  std::vector<uint8_t> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i)
    expected[i] = static_cast<uint8_t>(i * 37 + 11);

  // The source is filled and waited for before the copy, so the copy itself
  // is the only thing that could touch the source device's shared context.
  BufferHandle sourceHost;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kBytes, /*pinned_host=*/true,
                                   sourceHost.receive()),
            M_DRIVER_OK)
      << lastError();
  void *sourceHostAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(sourceHost.get(), &sourceHostAddress),
            M_DRIVER_OK);
  std::memcpy(sourceHostAddress, expected.data(), kBytes);
  BufferHandle source;
  ASSERT_EQ(M_DriverBufferAllocate(first.get(), kBytes, /*pinned_host=*/false,
                                   source.receive()),
            M_DRIVER_OK)
      << lastError();
  void *sourceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(source.get(), &sourceAddress), M_DRIVER_OK);
  StreamHandle producer;
  ASSERT_EQ(M_DriverStreamCreate(first.get(), producer.receive()), M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverCopyHostToDevice(producer.get(), sourceAddress,
                                     sourceHostAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(producer.get()), M_DRIVER_OK)
      << lastError();

  StreamHandle copier;
  ASSERT_EQ(M_DriverStreamCreate(second.get(), copier.receive()), M_DRIVER_OK)
      << lastError();
  BufferHandle destination;
  ASSERT_EQ(M_DriverBufferAllocate(second.get(), kBytes, /*pinned_host=*/false,
                                   destination.receive()),
            M_DRIVER_OK)
      << lastError();
  void *destinationAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(destination.get(), &destinationAddress),
            M_DRIVER_OK);
  BufferHandle landing;
  ASSERT_EQ(M_DriverBufferAllocate(second.get(), kBytes, /*pinned_host=*/true,
                                   landing.receive()),
            M_DRIVER_OK)
      << lastError();
  void *landingAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(landing.get(), &landingAddress), M_DRIVER_OK);

  // An error another user of the source device's shared context recorded
  // must neither fail the unordered copy nor be consumed by it.
  ErrorOrSuccess planted = plantPendingError(*sourceContext);
  ASSERT_FALSE(planted.isError()) << planted.getError();
  ASSERT_EQ(M_DriverCopyPeerToPeerUnordered(copier.get(), second.get(),
                                            destinationAddress, first.get(),
                                            sourceAddress, kBytes),
            M_DRIVER_OK)
      << "the unordered copy must not take the source context's pending "
         "error: "
      << lastError();
  ASSERT_EQ(M_DriverCopyDeviceToHost(copier.get(), landingAddress,
                                     destinationAddress, kBytes),
            M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(copier.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(std::memcmp(landingAddress, expected.data(), kBytes), 0);

  // The error is still pending on the source device, so its next
  // synchronous call reports it, and only then does the context run clean.
  EXPECT_EQ(M_DriverDeviceSynchronize(first.get()), M_DRIVER_RUNTIME_ERROR)
      << "the pending error must still be there for the next synchronous "
         "call to report";
  EXPECT_FALSE(lastError().empty());
  ASSERT_EQ(M_DriverDeviceSynchronize(first.get()), M_DRIVER_OK) << lastError();
#endif
}

} // namespace M::Driver
