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

#ifndef TARGET_LIB_METAL_CONTEXT_H
#define TARGET_LIB_METAL_CONTEXT_H

#include "Device.h"
#include "Driver.h"
#include "MetalCpp.h"
#include "Target/lib/Metal/Common.h"

#include <map>
#include <mutex>

struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device) : device(device) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  DriverResult<M_driver_memory_t> alloc(uint64_t size);
  M_driver_status_t free(M_driver_memory_t memory);

  M_driver_status_t copyToDeviceSync(M_driver_memory_t dst, uint64_t dstOffset,
                                     const void *src, uint64_t size);
  M_driver_status_t copyFromDeviceSync(void *dst, M_driver_memory_t src,
                                       uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size);
  M_driver_status_t setMemorySync(M_driver_memory_t dst, uint64_t dstOffset,
                                  uint64_t size, uint8_t value);
  M_driver_status_t fillSync(M_driver_memory_t dst, uint64_t dstOffset,
                             uint64_t size, uint64_t value, uint64_t valueSize);

  DriverResult<M_driver_memory_t> memoryWrap(uint64_t address, uint64_t size,
                                             bool owning);
  M_driver_status_t memoryUnwrap(M_driver_memory_t memory);

  void trackBuffer(void *ptr, NS::SharedPtr<MTL::Buffer> buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    allocations[reinterpret_cast<uint64_t>(ptr)] = buffer;
  }

  void untrackBuffer(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    // The map's `SharedPtr` is the buffer's sole owner, so erasing the entry
    // releases it exactly once; handles hold a non-owning raw pointer and must
    // not release it themselves.
    allocations.erase(reinterpret_cast<uint64_t>(ptr));
  }

  MTL::Buffer *getTrackedBuffer(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(reinterpret_cast<uint64_t>(ptr));
    if (it != allocations.end())
      return it->second.get();
    return nullptr;
  }

  bool isTrackedBuffer(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    return allocations.find(reinterpret_cast<uint64_t>(ptr)) !=
           allocations.end();
  }

  DriverResult<MTL::Buffer *>
  getTrackedBufferFromGPUAddress(MTL::GPUAddress gpuAddress) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &[_, buffer] : allocations) {
      if (isGPUAddressWithin(buffer.get(), gpuAddress))
        return buffer.get();
    }
    return tl::unexpected(device->driver->status.create(
        "MTL::Buffer for GPU address 0x" + Twine::utohexstr(gpuAddress) +
        " not found"));
  }

  bool isGPUAddressWithin(MTL::Buffer *buffer, MTL::GPUAddress gpuAddress) {
    auto begin = buffer->gpuAddress();
    auto end = begin + buffer->length();
    return gpuAddress >= begin && gpuAddress < end;
  }

  struct BufferWithOffset {
    MTL::Buffer *buffer;
    size_t offset;

    bool isValid() { return buffer != nullptr; }
  };

  BufferWithOffset getBufferWithOffset(const void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    uintptr_t ptrAddress = reinterpret_cast<uintptr_t>(ptr);
    // Find first allocation with base address > addr
    auto it = allocations.upper_bound(ptrAddress);
    if (it != allocations.begin()) {
      --it; // Step back to candidate allocation
      auto buffer = it->second;
      auto bufferAddress = uintptr_t(buffer->contents());
      if (ptrAddress >= bufferAddress &&
          ptrAddress < bufferAddress + buffer->length()) {
        return {buffer.get(), static_cast<size_t>(ptrAddress - bufferAddress)};
      }
    }
    return {nullptr, 0};
  }

  std::mutex mutex;
  M_driver_device_t device;
  std::map<uint64_t, NS::SharedPtr<MTL::Buffer>> allocations;
};

#endif // TARGET_LIB_METAL_CONTEXT_H
