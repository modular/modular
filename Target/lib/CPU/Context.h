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

#ifndef TARGET_LIB_CPU_CONTEXT_H
#define TARGET_LIB_CPU_CONTEXT_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "HAL/M_driver_common.h"

namespace AsyncRT = M::AsyncRT;

/// A CPU context owns the AsyncRT CPUDevice that backs all memory allocations
/// and worker-thread execution for queues created from it.
struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device,
                            AsyncRT::CPUDeviceRef cpuDevice)
      : device(device), cpuDevice(std::move(cpuDevice)) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  AsyncRT::CPUDevice &cpu() const { return *cpuDevice; }
  AsyncRT::Allocator *allocator() const { return cpuDevice->getAllocator(); }

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

  M_driver_device_t device;
  AsyncRT::CPUDeviceRef cpuDevice;
};

#endif // TARGET_LIB_CPU_CONTEXT_H
