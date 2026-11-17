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
// Tests for the chunked pinned host regions of the driver C API
// (`M_DriverHostRegion*` in MLRT/Driver/CAPI/DriverCAPI.h).
//
// A binary of its own because driver-allocated chunks on HIP need
// HSA_USERPTR_FOR_PAGED_MEM=0 set before the process starts, which changes how
// HIP serves every other pinned allocation in the process.
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

#include <unistd.h>

// The build defines this only under the HAL device context, and the test is
// compiled with every undefined macro in an #if treated as an error, so the
// classic build spells out the zero.
#ifndef MODULAR_HAL_DEVICE_CONTEXT
#define MODULAR_HAL_DEVICE_CONTEXT 0
#endif

using namespace capi_test;

namespace {

/// A host region that releases itself at scope exit.
class RegionHandle {
public:
  RegionHandle() = default;
  RegionHandle(const RegionHandle &) = delete;
  RegionHandle &operator=(const RegionHandle &) = delete;
  ~RegionHandle() { release(); }

  M_DriverHostRegion **receive() {
    release();
    return &handle_;
  }

  M_DriverHostRegion *get() const { return handle_; }

private:
  // The destroy calls refuse a null handle, so an empty owner skips them.
  void release() {
    if (handle_ != nullptr)
      M_DriverHostRegionDestroy(handle_);
    handle_ = nullptr;
  }

  M_DriverHostRegion *handle_ = nullptr;
};

size_t pageSize() { return static_cast<size_t>(sysconf(_SC_PAGE_SIZE)); }

/// One and a half pages, so a row is not a page multiple and the chunk size
/// has to round to both.
size_t rowBytes() { return pageSize() + pageSize() / 2; }

/// Seven rows: with chunks of two rows (three pages), four chunks, the last
/// holding a single row.
constexpr uint64_t kRows = 7;

struct Layout {
  uint64_t count = 0;
  uint64_t chunkBytes = 0;
  std::vector<uint8_t *> bases;
};

Layout layoutOf(const M_DriverHostRegion *region) {
  Layout layout;
  EXPECT_EQ(M_DriverHostRegionChunkCount(region, &layout.count), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(M_DriverHostRegionChunkBytes(region, &layout.chunkBytes),
            M_DRIVER_OK)
      << lastError();
  for (uint64_t i = 0; i < layout.count; ++i) {
    void *base = nullptr;
    EXPECT_EQ(M_DriverHostRegionChunkAddress(region, i, &base), M_DRIVER_OK)
        << lastError();
    layout.bases.push_back(static_cast<uint8_t *>(base));
  }
  return layout;
}

/// Sends a pattern from the last chunk to the device and back into the first,
/// so every chunk boundary the copy would cross if the layout were wrong is
/// between the two.
void roundTripsThroughTheDevice(const M_DriverDevice *device,
                                const Layout &layout, uint64_t lastBytes) {
  StreamHandle stream;
  ASSERT_EQ(M_DriverStreamCreate(device, stream.receive()), M_DRIVER_OK)
      << lastError();
  BufferHandle buffer;
  ASSERT_EQ(M_DriverBufferAllocate(device, lastBytes, /*pinned_host=*/false,
                                   buffer.receive()),
            M_DRIVER_OK)
      << lastError();
  void *deviceAddress = nullptr;
  ASSERT_EQ(M_DriverBufferPointer(buffer.get(), &deviceAddress), M_DRIVER_OK);
  ASSERT_EQ(M_DriverStreamWaitForDevice(stream.get(), device), M_DRIVER_OK)
      << lastError();

  uint8_t *last = layout.bases.back();
  uint8_t *first = layout.bases.front();
  for (uint64_t i = 0; i < lastBytes; ++i)
    last[i] = static_cast<uint8_t>(i * 7 + 3);
  std::memset(first, 0, lastBytes);

  ASSERT_EQ(
      M_DriverCopyHostToDevice(stream.get(), deviceAddress, last, lastBytes),
      M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(
      M_DriverCopyDeviceToHost(stream.get(), first, deviceAddress, lastBytes),
      M_DRIVER_OK)
      << lastError();
  ASSERT_EQ(M_DriverStreamSynchronize(stream.get()), M_DRIVER_OK)
      << lastError();
  EXPECT_EQ(std::memcmp(first, last, lastBytes), 0)
      << "the bytes sent from the last chunk must land in the first";
}

bool makeAccelerator(DeviceHandle &device) {
  const M_DriverStatus status =
      M_DriverAcceleratorDeviceCreate(0, device.receive());
  EXPECT_EQ(status, M_DRIVER_OK) << lastError();
  return status == M_DRIVER_OK;
}

} // namespace

TEST(DriverCApiHostRegion, RejectsNullHandlesAndTheCpuDevice) {
  M_DriverHostRegion *region = nullptr;
  EXPECT_EQ(M_DriverHostRegionCreate(nullptr, 4096, 4096, 0, false, false, 0,
                                     &region),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(region, nullptr);

  DeviceHandle cpu;
  ASSERT_TRUE(makeCpuDevice(cpu));
  EXPECT_EQ(M_DriverHostRegionCreate(cpu.get(), 4096, 4096, 0, false, false, 0,
                                     nullptr),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverHostRegionCreate(cpu.get(), 4096, 4096, 0, false, false, 0,
                                     &region),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(region, nullptr);
  EXPECT_FALSE(lastError().empty());

  uint64_t value = 0;
  void *address = nullptr;
  EXPECT_EQ(M_DriverHostRegionDestroy(nullptr), M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverHostRegionChunkCount(nullptr, &value),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverHostRegionChunkBytes(nullptr, &value),
            M_DRIVER_INVALID_ARGUMENT);
  EXPECT_EQ(M_DriverHostRegionChunkAddress(nullptr, 0, &address),
            M_DRIVER_INVALID_ARGUMENT);
}

TEST(DriverCApiHostRegion, RejectsSizesThatBreakTheRowRules) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "host regions need an accelerator";
  DeviceHandle device;
  ASSERT_TRUE(makeAccelerator(device));

  const uint64_t row = rowBytes();
  struct Case {
    const char *what;
    uint64_t byteSize;
    uint64_t rowBytes;
  };
  for (const Case &c :
       {Case{"a zero size", 0, row}, Case{"zero row bytes", row * kRows, 0},
        Case{"a size that is not whole rows", row * kRows + 1, row}}) {
    M_DriverHostRegion *region = nullptr;
    EXPECT_EQ(M_DriverHostRegionCreate(device.get(), c.byteSize, c.rowBytes, 0,
                                       /*driver_alloc=*/false,
                                       /*mlock_pages=*/false, 0, &region),
              M_DRIVER_INVALID_ARGUMENT)
        << c.what;
    EXPECT_EQ(region, nullptr) << c.what;
  }

  // Locking applies to the registered mapping only. The HAL build refuses
  // driver allocation before it looks at the other arguments.
  M_DriverHostRegion *region = nullptr;
  EXPECT_EQ(M_DriverHostRegionCreate(device.get(), row * kRows, row, 0,
                                     /*driver_alloc=*/true,
                                     /*mlock_pages=*/true, 0, &region),
            MODULAR_HAL_DEVICE_CONTEXT ? M_DRIVER_NOT_SUPPORTED
                                       : M_DRIVER_INVALID_ARGUMENT)
      << lastError();
  EXPECT_EQ(region, nullptr);
}

TEST(DriverCApiHostRegion, RegisteredRegionChunksWholeRowsAndCopies) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "host regions need an accelerator";
  DeviceHandle device;
  ASSERT_TRUE(makeAccelerator(device));

  const uint64_t row = rowBytes();
  const uint64_t byteSize = row * kRows;
  // Rounded down to whole rows and pages: three pages, two rows.
  const uint64_t requested = 4 * pageSize();
  const uint64_t expectedChunk = 3 * pageSize();

  // No touch threads beyond the default, one, and more than there are chunks.
  for (uint32_t threads : {0u, 1u, 64u}) {
    SCOPED_TRACE("touch_threads=" + std::to_string(threads));
    RegionHandle region;
    const M_DriverStatus status = M_DriverHostRegionCreate(
        device.get(), byteSize, row, requested, /*driver_alloc=*/false,
        /*mlock_pages=*/false, threads, region.receive());
#if MODULAR_HAL_DEVICE_CONTEXT
    EXPECT_EQ(status, M_DRIVER_NOT_SUPPORTED);
    GTEST_SKIP() << "the HAL shim registers no host memory: " << lastError();
#endif
    ASSERT_EQ(status, M_DRIVER_OK) << lastError();

    const Layout layout = layoutOf(region.get());
    EXPECT_EQ(layout.chunkBytes, expectedChunk);
    EXPECT_EQ(layout.chunkBytes % row, 0u);
    ASSERT_EQ(layout.count, (byteSize + expectedChunk - 1) / expectedChunk);
    // One mapping, so the chunks are back to back.
    for (uint64_t i = 1; i < layout.count; ++i)
      EXPECT_EQ(layout.bases[i], layout.bases[0] + i * layout.chunkBytes);

    void *past = nullptr;
    EXPECT_EQ(M_DriverHostRegionChunkAddress(region.get(), layout.count, &past),
              M_DRIVER_INVALID_ARGUMENT);
    EXPECT_EQ(M_DriverHostRegionChunkAddress(region.get(), 0, nullptr),
              M_DRIVER_INVALID_ARGUMENT);

    roundTripsThroughTheDevice(device.get(), layout,
                               byteSize -
                                   (layout.count - 1) * layout.chunkBytes);
  }
}

TEST(DriverCApiHostRegion, DefaultChunkSizeHoldsASmallRegionInOneChunk) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "host regions need an accelerator";
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim registers no host memory";
#endif
  DeviceHandle device;
  ASSERT_TRUE(makeAccelerator(device));

  // 0 selects the default; UINT64_MAX is the caller's way to ask for one chunk.
  const uint64_t row = rowBytes();
  for (uint64_t chunkBytes : {uint64_t{0}, UINT64_MAX}) {
    SCOPED_TRACE("chunk_bytes=" + std::to_string(chunkBytes));
    RegionHandle region;
    ASSERT_EQ(M_DriverHostRegionCreate(device.get(), row * kRows, row,
                                       chunkBytes, /*driver_alloc=*/false,
                                       /*mlock_pages=*/false, 0,
                                       region.receive()),
              M_DRIVER_OK)
        << lastError();
    const Layout layout = layoutOf(region.get());
    EXPECT_EQ(layout.count, 1u);
    EXPECT_GE(layout.chunkBytes, row * kRows);
  }
}

TEST(DriverCApiHostRegion, LockedRegionCopies) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "host regions need an accelerator";
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim registers no host memory";
#endif
  DeviceHandle device;
  ASSERT_TRUE(makeAccelerator(device));

  const uint64_t row = rowBytes();
  const uint64_t byteSize = row * kRows;
  RegionHandle region;
  const M_DriverStatus status = M_DriverHostRegionCreate(
      device.get(), byteSize, row, 4 * pageSize(), /*driver_alloc=*/false,
      /*mlock_pages=*/true, 0, region.receive());
  const std::string reason = lastError();
  if (status == M_DRIVER_OUT_OF_MEMORY &&
      reason.find("mlock") != std::string::npos)
    GTEST_SKIP() << "this host's RLIMIT_MEMLOCK is below the region: "
                 << reason;
  ASSERT_EQ(status, M_DRIVER_OK) << reason;

  const Layout layout = layoutOf(region.get());
  ASSERT_GT(layout.count, 1u);
  roundTripsThroughTheDevice(device.get(), layout,
                             byteSize - (layout.count - 1) * layout.chunkBytes);
}

TEST(DriverCApiHostRegion, DriverAllocatedRegionChunksWholeRowsAndCopies) {
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "host regions need an accelerator";
  DeviceHandle device;
  ASSERT_TRUE(makeAccelerator(device));

  const uint64_t row = rowBytes();
  const uint64_t byteSize = row * kRows;
  RegionHandle region;
  const M_DriverStatus status = M_DriverHostRegionCreate(
      device.get(), byteSize, row, 4 * pageSize(), /*driver_alloc=*/true,
      /*mlock_pages=*/false, 0, region.receive());
#if MODULAR_HAL_DEVICE_CONTEXT
  EXPECT_EQ(status, M_DRIVER_NOT_SUPPORTED) << lastError();
  EXPECT_EQ(region.get(), nullptr);
  return;
#endif
  char api[32] = {};
  ASSERT_NE(M_DriverDeviceApi(device.get(), api, sizeof(api)), 0u);
  if (status != M_DRIVER_OK && std::strcmp(api, "metal") == 0)
    GTEST_SKIP() << "Metal allocates no driver host chunks: " << lastError();
  ASSERT_EQ(status, M_DRIVER_OK) << lastError();

  const Layout layout = layoutOf(region.get());
  EXPECT_EQ(layout.chunkBytes, 3 * pageSize());
  ASSERT_EQ(layout.count,
            (byteSize + layout.chunkBytes - 1) / layout.chunkBytes);
  for (uint8_t *base : layout.bases)
    EXPECT_NE(base, nullptr);
  roundTripsThroughTheDevice(device.get(), layout,
                             byteSize - (layout.count - 1) * layout.chunkBytes);
}

namespace M::Driver {

// In the driver's namespace for its types.
TEST(DriverCApiHostRegion, CreateAndDestroyLeaveAPendingErrorInPlace) {
#if MODULAR_HAL_DEVICE_CONTEXT
  GTEST_SKIP() << "the HAL shim registers no host memory";
#else
  if (M_DriverAcceleratorCount() == 0)
    GTEST_SKIP() << "host regions need an accelerator";

  DeviceHandle device;
  ASSERT_EQ(M_DriverAcceleratorDeviceCreate(0, device.receive()), M_DRIVER_OK)
      << lastError();
  // The same process-wide context the handle holds, reached the way the
  // driver's own callers reach it.
  ErrorOr<DeviceContextRef> shared =
      DeviceContext::getOrCreate(Device::gpuAPI, 0);
  ASSERT_FALSE(shared.isError()) << shared.getError();
  DeviceContextRef context = shared.takeValue();

  ErrorOrSuccess planted = plantPendingError(*context);
  ASSERT_FALSE(planted.isError()) << planted.getError();

  const uint64_t row = rowBytes();
  M_DriverHostRegion *region = nullptr;
  ASSERT_EQ(M_DriverHostRegionCreate(device.get(), row * kRows, row,
                                     4 * pageSize(), /*driver_alloc=*/false,
                                     /*mlock_pages=*/false, 0, &region),
            M_DRIVER_OK)
      << "creating must not trip on the shared context's error: "
      << lastError();
  ASSERT_EQ(M_DriverHostRegionDestroy(region), M_DRIVER_OK) << lastError();

  EXPECT_EQ(M_DriverDeviceSynchronize(device.get()), M_DRIVER_RUNTIME_ERROR)
      << "the pending error must still be there for the next synchronous "
         "call to report";
  EXPECT_FALSE(lastError().empty());
#endif
}

} // namespace M::Driver
