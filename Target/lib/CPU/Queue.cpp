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
// This file implements queue operations for the CPU HAL plugin. All work is
// sequenced on an AsyncRT chain and executed on the CPUDevice's worker
// threads, a direct port of M::Driver::CpuDeviceContext's stream semantics.
//
//===----------------------------------------------------------------------===//

#include "Queue.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Support/Chain.h"
#include "Common.h"
#include "Context.h"
#include "Device.h"
#include "Event.h"
#include "HAL/M_driver_api.h"
#include "Memory.h"

#include <cstring>
#include <new>
#include <tuple>

M_driver_queue_t M_driver_queue::create(M_driver_context_t context) {
  // Seed the chain from the device's ready chain, mirroring
  // CpuDeviceContext's constructor.
  M_driver_queue_t queue = new (std::nothrow)
      M_driver_queue(context, context->cpu().getReadyChain().copy());
  if (queue != nullptr)
    context->device->registerQueue(queue);
  return queue;
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  // Wait for pending work to finish, mirroring ~CpuDeviceContext. The queue
  // stays registered until then, so a concurrent free still waits on it.
  if (!queue->streamChain.isReady())
    queue->context->cpu().getWorkQueue()->await({&queue->streamChain, 1});
  queue->context->device->unregisterQueue(queue);
  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

// Enqueue work with stream-like semantics: keep a chain to the last enqueued
// function so the next task awaits it.
M_driver_status_t M_driver_queue::enqueueWork(AsyncRT::TaskFunction &&work) {
  auto eventChain =
      AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(context->cpu());

  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  streamChain.andThenAsync(
      [work = std::move(work), eventChain = eventChain.copy()]() mutable {
        work();
        std::move(eventChain).emplace();
      });
  streamChain = std::move(eventChain);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst,
                                               uint64_t dstOffset,
                                               const void *src, uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  void *dstPtr = static_cast<uint8_t *>(dst->ptr) + dstOffset;
  return enqueueWork([=]() { memcpy(dstPtr, src, size); });
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t srcOffset,
                                                 uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  const void *srcPtr = static_cast<const uint8_t *>(src->ptr) + srcOffset;
  return enqueueWork([=]() { memcpy(dst, srcPtr, size); });
}

M_driver_status_t M_driver_queue::copyIntraDevice(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  M_driver_memory_t src,
                                                  uint64_t srcOffset,
                                                  uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  void *dstPtr = static_cast<uint8_t *>(dst->ptr) + dstOffset;
  const void *srcPtr = static_cast<const uint8_t *>(src->ptr) + srcOffset;
  return enqueueWork([=]() { memcpy(dstPtr, srcPtr, size); });
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t offset, uint64_t size,
                                            uint8_t value) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  void *dstPtr = static_cast<uint8_t *>(dst->ptr) + offset;
  return enqueueWork([=]() { memset(dstPtr, value, size); });
}

M_driver_status_t M_driver_queue::fillMemory(M_driver_memory_t dst,
                                             uint64_t offset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  if (valueSize != 1 && valueSize != 2 && valueSize != 4 && valueSize != 8)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size % valueSize != 0)
    return M_DRIVER_STATUS_INVALID_ARG;

  void *dstPtr = static_cast<uint8_t *>(dst->ptr) + offset;
  return enqueueWork([=]() {
    // `dstPtr` (dst->ptr + offset) is not guaranteed to be aligned to
    // `valueSize` -- the fill contract only constrains `size`, not `offset`.
    // Store via memcpy rather than casting to uintN_t* and dereferencing, which
    // would be undefined behavior on a misaligned pointer (and trips
    // -fsanitize=alignment). This mirrors the alignment-safe M::fillHomogeneous
    // path used by CpuDeviceContext.
    auto *bytes = static_cast<uint8_t *>(dstPtr);
    switch (valueSize) {
    case 1:
      memset(bytes, static_cast<uint8_t>(value), size);
      break;
    case 2: {
      uint16_t v16 = static_cast<uint16_t>(value);
      for (size_t i = 0; i < size; i += sizeof(v16))
        memcpy(bytes + i, &v16, sizeof(v16));
      break;
    }
    case 4: {
      uint32_t v32 = static_cast<uint32_t>(value);
      for (size_t i = 0; i < size; i += sizeof(v32))
        memcpy(bytes + i, &v32, sizeof(v32));
      break;
    }
    case 8: {
      for (size_t i = 0; i < size; i += sizeof(value))
        memcpy(bytes + i, &value, sizeof(value));
      break;
    }
    }
  });
}

M_driver_status_t M_driver_queue::freeAsync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  // The Runtime's allocator outlives any work enqueued on its work queue, so
  // capturing a raw pointer (rather than a CPUDeviceRef) is safe.
  auto *allocator = context->allocator();
  void *ptr = memory->ptr;
  uint64_t size = memory->size;
  bool owning = memory->owning;
  delete memory;
  // A non-owning handle wraps memory this plugin did not allocate; drop the
  // handle without touching the region.
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  return enqueueWork(
      [allocator, ptr, size]() { allocator->deallocateBytes(ptr, size); });
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  event->value = streamChain.copy();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t numEvents) {
  if (numEvents == 0)
    return M_DRIVER_STATUS_SUCCESS;

  std::lock_guard<std::mutex> streamLock{streamChainMutex};
  // Make subsequent work await each event, mirroring
  // CpuDeviceContext::enqueueWaitFor.
  for (uint32_t i = 0; i < numEvents; ++i) {
    auto eventChain =
        AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(context->cpu());
    AsyncRT::andThenAsync(
        std::make_tuple(streamChain.copy(), events[i]->value.copy()),
        [eventChain = eventChain.copy()](auto, auto) mutable {
          std::move(eventChain).emplace();
        });
    streamChain = std::move(eventChain);
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  AsyncRT::AnyAsyncValueRef head;
  {
    std::lock_guard<std::mutex> streamLock{streamChainMutex};
    head = streamChain.copy();
  }
  context->cpu().getWorkQueue()->await({&head, 1});
  return M_DRIVER_STATUS_SUCCESS;
}

// ===-------------------------------------------------------------------=== //
// C API dispatch
// ===-------------------------------------------------------------------=== //

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (queue == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_queue_t result = M_driver_queue::create(context);
  if (result == nullptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  *queue = result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue_destroy([[maybe_unused]] M_driver_context_t context,
                       M_driver_queue_t queue) {
  return M_driver_queue::destroy(queue);
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return queue->copyToDevice(dst->memory, dst->offset, src, dst->size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (src == nullptr || src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (dst == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return queue->copyFromDevice(dst, src->memory, src->offset, src->size);
}

M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || src == nullptr || dst->memory == nullptr ||
      src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyIntraDevice(dst->memory, dst->offset, src->memory,
                                src->offset, dst->size);
}

M_driver_status_t M_driver_queue_copy_inter_device(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || src == nullptr || dst->memory == nullptr ||
      src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src_context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return queue->copyIntraDevice(dst->memory, dst->offset, src->memory,
                                src->offset, dst->size);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->setMemory(dst->memory, dst->offset, dst->size, value);
}

M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->fillMemory(dst->memory, dst->offset, dst->size, value,
                           value_size);
}

M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t /*function*/,
                       M_driver_queue_execute_config_t * /*config*/,
                       void ** /*args*/, uint64_t * /*arg_sizes*/,
                       uint32_t /*num_args*/) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  // CPU devices do not launch GPU kernels.
  return M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED;
}

M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return queue->recordEvent(event);
}

M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
                                                 M_driver_event_t *events,
                                                 uint32_t num_events) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (events == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return queue->waitForEvents(events, num_events);
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->synchronize();
}

M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (is_stream == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // Each queue is a totally-ordered async chain, i.e. an in-order stream.
  *is_stream = true;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

// Native stream handles and host-function callbacks are not supported on the
// CPU queue yet.

M_driver_status_t
M_driver_queue_launch_host_func([[maybe_unused]] M_driver_queue_t queue,
                                [[maybe_unused]] void (*fn)(void *),
                                [[maybe_unused]] void *user_data) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_queue_wait_value64([[maybe_unused]] M_driver_queue_t queue,
                            [[maybe_unused]] uint64_t device_address,
                            [[maybe_unused]] uint64_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
