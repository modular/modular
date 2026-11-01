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

#ifndef TARGET_LIB_CUDA_QUEUE_H
#define TARGET_LIB_CUDA_QUEUE_H

#include "Error.h"
#include "Memory.h"

#include <cuda.h>

struct M_driver_context;

struct M_driver_queue {
  M_driver_context &context;
  CUstream stream;

  [[nodiscard]] static DriverResult<M_driver_queue_t>
  create(M_driver_context &ctx);

  static M_driver_status_t destroy(M_driver_queue_t queue);

  DriverResult<M_driver_memory_t> allocAsync(uint64_t size);
  M_driver_status_t freeAsync(M_driver_memory_t memory);
  M_driver_status_t copyToDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                 const void *src, uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                    M_driver_memory_t src, uint64_t srcOffset,
                                    uint64_t size);
  M_driver_status_t copyInterDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                    M_driver_memory_t src, uint64_t srcOffset,
                                    uint64_t size,
                                    M_driver_context_t srcContext);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t dstOffset,
                              uint64_t size, uint8_t value);
  M_driver_status_t fillMemory(M_driver_memory_t dst, uint64_t dstOffset,
                               uint64_t size, uint64_t value,
                               uint64_t valueSize);

  M_driver_status_t synchronize();

  M_driver_status_t recordEvent(M_driver_event_t event);
  M_driver_status_t waitForEvents(M_driver_event_t *events,
                                  uint32_t num_events);

  M_driver_status_t launchHostFunc(void (*fn)(void *), void *user_data);
  M_driver_status_t waitValue64(uint64_t device_address, uint64_t value);

  explicit M_driver_queue(M_driver_context &context, CUstream stream)
      : context(context), stream(stream) {}
};

#endif // TARGET_LIB_CUDA_QUEUE_H
