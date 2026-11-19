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

#include "AllocatorBacking.h"
#include "AllocatorContext.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

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

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_DEVICEALLOCATOR_H
