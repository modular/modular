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
#include "Driver.h"
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "Queue.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

using M::Target::HIP::hipErrorStatus;

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (queue == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  DriverResult<M_driver_queue_t> result = M_driver_queue::create(*context);
  if (!result)
    return result.error();
  *queue = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue) {
  return M_driver_queue::destroy(queue);
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->copyToDevice(dst->memory, dst->offset, src, dst->size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->copyFromDevice(dst, src->memory, src->offset, src->size);
}

M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyIntraDevice(dst->memory, dst->offset, src->memory,
                                src->offset, dst->size);
}

M_driver_status_t M_driver_queue_copy_inter_device(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src_context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return queue->copyInterDevice(dst->memory, dst->offset, src->memory,
                                src->offset, dst->size, src_context);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->setMemory(dst->memory, dst->offset, dst->size, value);
}

M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->fillMemory(dst->memory, dst->offset, dst->size, value,
                           value_size);
}

M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (config == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  M_driver_device &dev = queue->context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  auto &gpu = config->config.gpu;

  // Mirrors HIPDeviceContext: function launch attributes are not supported.
  if (gpu.num_attributes > 0)
    return M_DRIVER_STATUS_UNIMPLEMENTED;

  if (hipError_t r = hip.hipModuleLaunchKernel(
          function->function, gpu.grid.x, gpu.grid.y, gpu.grid.z, gpu.block.x,
          gpu.block.y, gpu.block.z, gpu.shared_mem_bytes, queue->stream, args,
          /*extra=*/nullptr);
      r != hipSuccess)
    return hipErrorStatus(
        &dev.getDriver(), r,
        llvm::formatv("hipModuleLaunchKernel grid=({0},{1},{2}) "
                      "block=({3},{4},{5}) smem={6}",
                      gpu.grid.x, gpu.grid.y, gpu.grid.z, gpu.block.x,
                      gpu.block.y, gpu.block.z, gpu.shared_mem_bytes)
            .str());

  return M_DRIVER_STATUS_SUCCESS;
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

M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (llvm::StringRef(property_name) == M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE) {
    *static_cast<uint64_t *>(value) = reinterpret_cast<uint64_t>(queue->stream);
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (is_stream == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_stream = true;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_launch_host_func(M_driver_queue_t queue,
                                                  void (*fn)(void *),
                                                  void *user_data) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->launchHostFunc(fn, user_data);
}

M_driver_status_t M_driver_queue_wait_value64(M_driver_queue_t queue,
                                              uint64_t device_address,
                                              uint64_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->waitValue64(device_address, value);
}
