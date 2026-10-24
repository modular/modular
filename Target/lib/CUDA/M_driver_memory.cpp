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

#include "Context.h"
#include "HAL/M_driver_api.h"
#include "Memory.h"
#include "Queue.h"

M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_memory_t> result = context->allocPinned(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_memory_alloc_sync([[maybe_unused]] M_driver_context_t context,
                           [[maybe_unused]] uint64_t size,
                           [[maybe_unused]] M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_memory_t> result = context->allocSync(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->freePinned(memory);
}

M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->freeSync(memory);
}

M_driver_status_t M_driver_memory_wrap(M_driver_context_t context,
                                       uint64_t address, uint64_t size,
                                       bool owning, M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_memory_t> result =
      context->wrapSync(address, size, owning);
  if (!result)
    return result.error();
  *memory = *result;
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

  DriverResult<M_driver_memory_t> result = queue->allocAsync(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
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
  return memory->getProperty(property_name, value);
}
