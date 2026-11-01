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
// This file implements memory management for the CPU HAL plugin. Allocations
// go through the AsyncRT CPUDevice allocator, mirroring
// CpuDeviceContext::memAlloc / memFree.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "Common.h"
#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_memory.h"
#include "Queue.h"

#include <new>

M_driver_status_t M_driver_memory::getProperty(llvm::StringRef name,
                                               void *value) const {
  if (name == M_DRIVER_MEMORY_PROPERTY_ADDRESS ||
      name == M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS) {
    *static_cast<uint64_t *>(value) = reinterpret_cast<uint64_t>(ptr);
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

namespace {

// Shared allocation/free helpers used by the pinned and sync entry points;
// on CPU both kinds of memory are plain host allocations.
M_driver_status_t allocImpl(M_driver_context_t context, uint64_t size,
                            M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size == 0) {
    *memory = nullptr;
    return M_DRIVER_STATUS_SUCCESS;
  }

  // Match CpuDeviceContext::memAlloc: alignment of 1 (the allocator applies
  // its own minimum alignment).
  void *ptr = context->allocator()->allocateBytes(size, /*alignment=*/1);
  if (ptr == nullptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;

  M_driver_memory_t result = new (std::nothrow) M_driver_memory{ptr, size};
  if (result == nullptr) {
    context->allocator()->deallocateBytes(ptr, size);
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  }
  *memory = result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t freeImpl(M_driver_context_t context,
                           M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  bool owning = memory->owning;
  void *ptr = memory->ptr;
  uint64_t size = memory->size;
  delete memory;
  // A non-owning handle wraps memory this plugin did not allocate, so it frees
  // only the handle above and leaves the region alone.
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  // Work already enqueued may still read or write the region. A caller can
  // drop its last reference to a buffer right after enqueueing a copy of it,
  // so wait for that work instead of freeing under it.
  // TODO(DRIV-579): This waits for every queue on the device, including
  // ones that never touched this memory, so a free can stall behind unrelated
  // long-running work. Remove it once resource lifetime management makes each
  // free wait only on the queued work that uses the buffer.
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef, 4> pending =
      context->device->pendingWork();
  if (!pending.empty())
    context->cpu().getWorkQueue()->await(pending);
  context->allocator()->deallocateBytes(ptr, size);
  return M_DRIVER_STATUS_SUCCESS;
}

} // namespace

M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory) {
  return allocImpl(context, size, memory);
}

M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
                                             uint64_t size,
                                             M_driver_memory_t *memory) {
  return allocImpl(context, size, memory);
}

M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  return freeImpl(context, memory);
}

M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  return freeImpl(context, memory);
}

M_driver_status_t M_driver_memory_wrap(M_driver_context_t context,
                                       uint64_t address, uint64_t size,
                                       bool owning, M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (address == 0)
    return M_DRIVER_STATUS_INVALID_ARG;

  // On CPU the device address is the host pointer, so wrapping is just
  // recording it. `owning=true` is only valid for an address from this
  // plugin's own allocator (e.g. one released with unwrap), so freeing it
  // routes back through the allocator with the retained size.
  M_driver_memory_t result = new (std::nothrow)
      M_driver_memory{reinterpret_cast<void *>(address), size, owning};
  if (result == nullptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  *memory = result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_unwrap(M_driver_context_t context,
                                         M_driver_memory_t memory,
                                         uint64_t *address) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (address == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  memory->owning = false;
  *address = reinterpret_cast<uint64_t>(memory->ptr);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // CPU allocation is synchronous; only the free is stream-ordered. This
  // matches CpuDeviceContext behavior.
  return allocImpl(queue->context, size, memory);
}

M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->freeAsync(memory);
}

M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
                                           const char *property_name,
                                           void *value) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return memory->getProperty(property_name, value);
}
