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
// This file declares what the legacy driver uses to serve a `MemoryManager`.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_DEVICEALLOCATOR_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_DEVICEALLOCATOR_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/MemoryManager/AllocatorBacking.h"
#include "MLRT/lib/Driver/MemoryManager/AllocatorContext.h"

#include <memory>
#include <string>

namespace M::Driver {

class MemoryManager;
class MemoryManagerFreeList;

AllocatorContext::AllocMode toAllocatorAllocMode(Device::AllocMode mode);
Device::MemInfo toDeviceMemInfo(const AllocatorMemInfo &info);

// Each method forwards to the `Device`.
class DeviceAllocatorChunks final : public AllocatorChunks {
public:
  explicit DeviceAllocatorChunks(Device &device) : device_(device) {}

  ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize, bool isHost) override;
  ErrorOrSuccess chunkFree(void *ptr, bool isHost) override;
  ErrorOr<AllocatorMemInfo> getMemoryInfo() const override;

private:
  Device &device_;
};

class DeviceAllocatorVmm final : public AllocatorVmm {
public:
  explicit DeviceAllocatorVmm(Device &device) : device_(device) {}

  ErrorOr<void *> vmmAddressReserve(size_t byteSize) override;
  ErrorOrSuccess vmmAddressFree(void *va, size_t byteSize) override;
  ErrorOr<uint64_t> vmmCreate(size_t byteSize) override;
  ErrorOrSuccess vmmRelease(uint64_t handle) override;
  ErrorOrSuccess vmmMapAndSetAccess(void *va, size_t byteSize,
                                    uint64_t handle) override;
  ErrorOrSuccess vmmUnmap(void *va, size_t byteSize) override;
  ErrorOrSuccess vmmMap(void *va, size_t byteSize, uint64_t handle) override;
  ErrorOrSuccess vmmSetAccess(void *va, size_t byteSize) override;

private:
  Device &device_;
};

// Builds a free list for `context`, with the redzones and poison that its
// `debug.device_allocator` config asks for.
std::unique_ptr<MemoryManagerFreeList>
createFreeListForContext(DeviceContext &context, MemoryManager &memoryManager,
                         MemoryManagerFreeList *parent, bool disableFlush,
                         std::string label);

// These factories configure a memory manager from defaults and the driver
// config. `memInfo` holds the device's total and free memory; an unset
// `memInfo.free` sizes the manager against `memInfo.total`. `memoryLimit`
// (bytes) caps the manager; -1 uses the default limit.
// `largeAllocAlignmentOverride` (bytes) is the start alignment given to large
// blocks, rounded up to a power of 2; 0 disables it and -1 takes the value
// from the config.
std::unique_ptr<MemoryManager>
createDeviceMemoryManager(Device &device, Device::MemInfo memInfo,
                          size_t alignment, int64_t memoryLimit = -1,
                          int64_t largeAllocAlignmentOverride = -1);
std::unique_ptr<MemoryManager> createHostMemoryManager(Device &device,
                                                       size_t alignment);
// A device memory manager with no budget, for devices that run without one.
std::unique_ptr<MemoryManager> createDisabledMemoryManager(Device &device,
                                                           size_t alignment);

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_DEVICEALLOCATOR_H
