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
#include "Device.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  DriverResult<M_driver_context_t> result = M_driver_context::create(*device);
  if (!result)
    return result.error();
  *context = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_driver_context::destroy(context);
}

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

M_driver_status_t M_driver_context_set_current(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->setCurrent();
}

M_driver_status_t M_driver_context_get_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t *driver_context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->getCurrentDriverContext(driver_context);
}

M_driver_status_t M_driver_context_set_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t driver_context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->setCurrentDriverContext(driver_context);
}

M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
                                               uint64_t *free,
                                               uint64_t *total) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->memoryInfo(free, total);
}

M_driver_status_t
M_driver_context_enable_peer_access(M_driver_context_t context,
                                    M_driver_context_t peer) {
  if (context == nullptr || peer == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->enablePeerAccess(*peer);
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
