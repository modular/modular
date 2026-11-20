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

#ifndef TARGET_LIB_HIP_CONTEXT_H
#define TARGET_LIB_HIP_CONTEXT_H

#include "Error.h"
#include "HAL/M_driver_memory.h"
#include "llvm/ADT/DenseMap.h"

struct M_driver_device;

struct M_driver_context {
private:
  M_driver_device &device;

public:
  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  // TODO: should we implicitly also dealloc during destroy?
  static M_driver_status_t destroy(M_driver_context_t ctx) {
    delete ctx;
    return M_DRIVER_STATUS_SUCCESS;
  }

  explicit M_driver_context(M_driver_device &device) : device(device) {}

  M_driver_device &getDevice() { return device; }

  DriverResult<M_driver_memory_t> allocPinned(uint64_t size);
  DriverResult<M_driver_memory_t> allocSync(uint64_t size);
  DriverResult<M_driver_memory_t> wrapSync(uint64_t address, uint64_t size,
                                           bool owning);
  M_driver_status_t setCurrent();
  M_driver_status_t
  getCurrentDriverContext(M_driver_native_handle_t *driver_context);
  M_driver_status_t
  setCurrentDriverContext(M_driver_native_handle_t driver_context);
  M_driver_status_t memoryInfo(uint64_t *free, uint64_t *total);
  M_driver_status_t enablePeerAccess(M_driver_context &peer);
  M_driver_status_t freePinned(M_driver_memory_t memory);
  M_driver_status_t freeSync(M_driver_memory_t memory);

  M_driver_status_t copyToDeviceSync(M_driver_memory_t dst, uint64_t dstOffset,
                                     const void *src, uint64_t size);
  M_driver_status_t copyFromDeviceSync(void *dst, M_driver_memory_t src,
                                       uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyInterDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size,
                                        M_driver_context_t srcContext);
  M_driver_status_t setMemorySync(M_driver_memory_t dst, uint64_t dstOffset,
                                  uint64_t size, uint8_t value);
  M_driver_status_t fillSync(M_driver_memory_t dst, uint64_t dstOffset,
                             uint64_t size, uint64_t value, uint64_t valueSize);
};

#endif // TARGET_LIB_HIP_CONTEXT_H
