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
// This file implements context lifecycle for the CPU HAL plugin. A context
// acquires the process-wide AsyncRT CPUDevice, exactly as
// M::Driver::CpuDeviceContext does in its constructor.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "Device.h"
#include "HAL/M_driver_context.h"
#include "Memory.h"

#include <cerrno>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>

M_driver_context_t M_driver_context::create(M_driver_device_t device) {
  // Get or create the AsyncRT CPUDevice. Request mainWillDonate=false since
  // the HAL caller drives work from its own thread, but allow reusing an
  // existing CPUDevice's options if one already exists.
  AsyncRT::CPUDeviceRef cpuDevice = AsyncRT::getOrCreateCPUDevice(
      AsyncRT::CPUDeviceSource::CPUDeviceContext,
      AsyncRT::CPUDeviceOptions().withMainWillNotDonate(),
      /*allowUsingExistingOptions=*/true);
  return new (std::nothrow) M_driver_context(device, std::move(cpuDevice));
}

void M_driver_context::destroy(M_driver_context_t context) { delete context; }

M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_context_t result = M_driver_context::create(device);
  if (result == nullptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  *context = result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  M_driver_context::destroy(context);
  return M_DRIVER_STATUS_SUCCESS;
}

// Synchronous copy methods. CPU memory is ordinary host memory, so every
// transfer is a plain blocking memcpy. A cross-device (inter) copy is likewise
// a memcpy: there is a single host address space, so `src_context` is
// irrelevant beyond being a valid handle.
M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  if (size > 0)
    memcpy(static_cast<char *>(dst->ptr) + dstOffset, src, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  if (size > 0)
    memcpy(dst, static_cast<char *>(src->ptr) + srcOffset, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(M_driver_memory_t dst,
                                                        uint64_t dstOffset,
                                                        M_driver_memory_t src,
                                                        uint64_t srcOffset,
                                                        uint64_t size) {
  if (size > 0)
    memcpy(static_cast<char *>(dst->ptr) + dstOffset,
           static_cast<char *>(src->ptr) + srcOffset, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyInterDeviceSync(
    M_driver_memory_t dst, uint64_t dstOffset, M_driver_memory_t src,
    uint64_t srcOffset, uint64_t size, M_driver_context_t /*srcContext*/) {
  if (size > 0)
    memcpy(static_cast<char *>(dst->ptr) + dstOffset,
           static_cast<char *>(src->ptr) + srcOffset, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setMemorySync(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  uint64_t size,
                                                  uint8_t value) {
  if (size > 0)
    memset(static_cast<char *>(dst->ptr) + dstOffset, value, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::fillSync(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;
  if (valueSize != 1 && valueSize != 2 && valueSize != 4 && valueSize != 8)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size % valueSize != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (valueSize == 1)
    return setMemorySync(dst, dstOffset, size, static_cast<uint8_t>(value));

  auto *bytes = static_cast<uint8_t *>(dst->ptr) + dstOffset;
  switch (valueSize) {
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
  return M_DRIVER_STATUS_SUCCESS;
}

// Synchronous copy C ABI: thin wrappers over the M_driver_context methods.
M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t context,
                                               M_driver_memory_view_t dst,
                                               const void *src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return context->copyToDeviceSync(dst->memory, dst->offset, src, dst->size);
}

M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t context,
                                                 void *dst,
                                                 M_driver_memory_view_t src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (src == nullptr || src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (dst == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return context->copyFromDeviceSync(dst, src->memory, src->offset, src->size);
}

M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t context,
                                                  M_driver_memory_view_t dst,
                                                  M_driver_memory_view_t src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || src == nullptr || dst->memory == nullptr ||
      src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return context->copyIntraDeviceSync(dst->memory, dst->offset, src->memory,
                                      src->offset, dst->size);
}

M_driver_status_t M_driver_copy_inter_device_sync(
    M_driver_context_t context, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) {
  if (context == nullptr || src_context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || src == nullptr || dst->memory == nullptr ||
      src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return context->copyInterDeviceSync(dst->memory, dst->offset, src->memory,
                                      src->offset, dst->size, src_context);
}

// The host has no driver-level current-context state to install or save;
// these mirror CpuDevice::getCurrentDriverContext / setCurrentDriverContext
// (a null handle, and a no-op restore).

M_driver_status_t M_driver_context_set_current(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_get_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t *driver_context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *driver_context = 0;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_set_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t /*driver_context*/) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
                                               uint64_t *free,
                                               uint64_t *total) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  // Mirrors CpuDevice::getMemoryInfo: total physical memory from sysconf, and
  // "free" approximated as total minus this process's peak resident set size.
  errno = 0;
  long numPages = sysconf(_SC_PHYS_PAGES);
  if (errno != 0)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  errno = 0;
  long pageSize = sysconf(_SC_PAGE_SIZE);
  if (errno != 0)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == -1)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

// ru_maxrss is kilobytes on Linux and bytes on macOS.
#ifdef __APPLE__
  const uint64_t maxrssMultiple = 1;
#else
  const uint64_t maxrssMultiple = 1024;
#endif
  uint64_t used = static_cast<uint64_t>(usage.ru_maxrss) * maxrssMultiple;
  uint64_t totalBytes =
      static_cast<uint64_t>(numPages) * static_cast<uint64_t>(pageSize);
  *total = totalBytes;
  *free = used < totalBytes ? totalBytes - used : 0;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_set_memory_sync(M_driver_context_t context,
                                           M_driver_memory_view_t dst,
                                           uint8_t value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return context->setMemorySync(dst->memory, dst->offset, dst->size, value);
}

M_driver_status_t M_driver_fill_sync(M_driver_context_t context,
                                     M_driver_memory_view_t dst, uint64_t value,
                                     uint64_t value_size) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return context->fillSync(dst->memory, dst->offset, dst->size, value,
                           value_size);
}
