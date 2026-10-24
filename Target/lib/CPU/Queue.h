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

#ifndef TARGET_LIB_CPU_QUEUE_H
#define TARGET_LIB_CPU_QUEUE_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "HAL/M_driver_common.h"
#include "HAL/M_driver_queue.h"

#include <cstdint>
#include <mutex>

namespace AsyncRT = M::AsyncRT;

/// A CPU queue is an ordered chain of async work items. Each enqueued
/// operation is sequenced after the previous one via `streamChain`, exactly
/// as M::Driver::CpuDeviceContext sequences work on its single stream. Because
/// the chain is totally ordered, the queue reports itself as a stream.
struct M_driver_queue {
  explicit M_driver_queue(M_driver_context_t context,
                          AsyncRT::AnyAsyncValueRef streamChain)
      : context(context), streamChain(std::move(streamChain)) {}

  [[nodiscard]] static M_driver_queue_t create(M_driver_context_t context);
  static M_driver_status_t destroy(M_driver_queue_t queue);

  M_driver_status_t copyToDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                 const void *src, uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                    M_driver_memory_t src, uint64_t srcOffset,
                                    uint64_t size);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t offset,
                              uint64_t size, uint8_t value);
  M_driver_status_t fillMemory(M_driver_memory_t dst, uint64_t offset,
                               uint64_t size, uint64_t value,
                               uint64_t valueSize);

  M_driver_status_t freeAsync(M_driver_memory_t memory);

  M_driver_status_t recordEvent(M_driver_event_t event);
  M_driver_status_t waitForEvents(M_driver_event_t *events, uint32_t numEvents);

  M_driver_status_t synchronize();

  M_driver_context_t context;

  // Chain of the last enqueued operation, which may already have completed.
  // Guarded by `streamChainMutex`.
  AsyncRT::AnyAsyncValueRef streamChain;
  std::mutex streamChainMutex;

private:
  // Enqueue `work` after the current chain tail, advancing the tail. Mirrors
  // CpuDeviceContext::enqueueWork.
  M_driver_status_t enqueueWork(AsyncRT::TaskFunction &&work);
};

#endif // TARGET_LIB_CPU_QUEUE_H
