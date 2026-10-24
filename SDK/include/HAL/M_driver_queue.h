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

#ifndef M_DRIVER_QUEUE_H
#define M_DRIVER_QUEUE_H

#include "M_driver_common.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a queue for the given context.
 *
 * @param[in] context The context to create the queue for.
 * @param[out] queue Pointer to store the newly created queue.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue);

typedef M_driver_status_t (*M_driver_queue_create_func_ptr_t)(
    M_driver_context_t context, M_driver_queue_t *queue);

/**
 * Destroy a previously created queue.
 *
 * @param[in] context The context the queue was created for.
 * @param[in] queue The queue to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue);

typedef M_driver_status_t (*M_driver_queue_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_queue_t queue);

/**
 * Copy data from host memory to device memory.
 *
 * @param[in] queue The queue to use for the copy operation.
 * @param[in] dst Destination device-memory view; `dst->size` bytes are copied
 *            starting at `dst->offset` into `dst->memory`.
 * @param[in] src Pointer to the source host memory.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src);

typedef M_driver_status_t (*M_driver_queue_copy_to_device_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_view_t dst, const void *src);

/**
 * Copy data from device memory to host memory.
 *
 * @param[in] queue The queue to use for the copy operation.
 * @param[in] dst Pointer to the destination host memory.
 * @param[in] src Source device-memory view; `src->size` bytes are copied
 *            starting at `src->offset` into `src->memory`.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src);

typedef M_driver_status_t (*M_driver_queue_copy_from_device_func_ptr_t)(
    M_driver_queue_t queue, void *dst, M_driver_memory_view_t src);

/**
 * Copy data between two memory regions on the same device.
 *
 * Both @p dst and @p src must reside on the device backing @p queue.
 * `dst->size` bytes are copied; @p src must be at least that large.
 *
 * @param[in] queue The queue to use for the copy operation.
 * @param[in] dst Destination device-memory view.
 * @param[in] src Source device-memory view.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src);

typedef M_driver_status_t (*M_driver_queue_copy_intra_device_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src);

/**
 * Copy data between memory regions on two different devices (a peer copy).
 *
 * @p dst must reside on the device backing @p queue; @p src resides on the
 * device backing @p src_context. `dst->size` bytes are copied; @p src must be
 * at least that large. The copy is enqueued on @p queue, which belongs to the
 * destination's context.
 *
 * @param[in] queue The queue to enqueue the copy on; belongs to @p dst's
 *            device.
 * @param[in] dst Destination device-memory view, on @p queue's device.
 * @param[in] src Source device-memory view, on @p src_context's device.
 * @param[in] src_context The context that owns @p src, used to resolve the
 *            peer (source) device.
 * @return Standard Status Protocol, or M_DRIVER_STATUS_UNIMPLEMENTED on
 *         backends without peer-copy support.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_inter_device(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context);

typedef M_driver_status_t (*M_driver_queue_copy_inter_device_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context);

/**
 * Set device memory to a repeated single byte.
 *
 * @param[in] queue The queue to use for the set operation.
 * @param[in] dst Destination device-memory view; `dst->size` bytes starting at
 *            `dst->offset` into `dst->memory` are set.
 * @param[in] value The byte value to set.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value);

typedef M_driver_status_t (*M_driver_queue_set_memory_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_view_t dst, uint8_t value);

/**
 * Fill device memory with a repeated `value_size`-byte value.
 *
 * @param[in] queue The queue to use for the fill operation.
 * @param[in] dst Destination device-memory view; `dst->size` bytes starting at
 *            `dst->offset` into `dst->memory` are filled. `dst->size` must be a
 *            multiple of `value_size`.
 * @param[in] value The value to fill with; the low `value_size` bytes are used.
 * @param[in] value_size The size of the value in bytes. Must be one of 1, 2, 4,
 *            or 8. A `value_size` of 1 is equivalent to
 *            M_driver_queue_set_memory.
 * @return Standard Status Protocol, or M_DRIVER_STATUS_INVALID_ARG if
 *         `value_size` is unsupported or `dst->size` is not a multiple of it.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size);

typedef M_driver_status_t (*M_driver_queue_fill_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_view_t dst, uint64_t value,
    uint64_t value_size);

/**
 * Struct representing a three dimensional launch configuration.
 */
typedef struct M_driver_dim_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} M_driver_dim_t;

/**
 * Struct representing a GPU launch configuration for executing an
 * M_driver_function_t. Used in M_driver_queue_execute() via
 * M_driver_queue_execute_config_t when targeting execution on a GPU device.
 */
typedef struct M_driver_queue_execute_config_gpu_t {
  M_driver_dim_t grid;       ///< The grid dimension, AKA ND-range
  M_driver_dim_t block;      ///< The block dimension, AKA work-group size.
  uint32_t shared_mem_bytes; ///< Shared memory size in bytes.
  void *attributes;          ///< Passthrough of CUlaunchAttributes
  uint32_t num_attributes;   ///< Number of elements pointed to by attributes.
} M_driver_queue_execute_config_gpu_t;

/** NPU - Temporary struct for NPU launch configuration. */
typedef struct M_driver_queue_execute_config_npu_t {
  uint32_t cuBitmask;
} M_driver_queue_execute_config_npu_t;

// Selects the launch config mode.
typedef enum M_driver_queue_execute_mode_t {
  M_DRIVER_QUEUE_EXECUTE_MODE_GPU, ///< Specify execution on a GPU device.
  M_DRIVER_QUEUE_EXECUTE_MODE_NPU, ///< Specify execution on a NPU device.
} M_driver_queue_execute_mode_t;

/**
 * Struct representing a function execution launch configuration, contains
 * the config union which is discriminated by the mode enum.
 */
typedef struct M_driver_queue_execute_config_t {
  M_driver_queue_execute_mode_t mode; ///< Enum discriminates the config union.
  union M_driver_queue_execute_config_union_t {
    /// Execution launch configuration for GPU, used when mode is
    /// M_DRIVER_QUEUE_EXECUTE_MODE_GPU.
    M_driver_queue_execute_config_gpu_t gpu;
    /// Execution launch configuration for NPU, used when mode is
    /// M_DRIVER_QUEUE_EXECUTE_MODE_NPU.
    M_driver_queue_execute_config_npu_t npu;
  } config; ///< The union discriminated by mode.
} M_driver_queue_execute_config_t;

/**
 * Execute a function on a device's queue.
 *
 * @param[in] queue The queue where the function will be executed.
 * @param[in] function The function which will be executed.
 * @param[in] config The function execution launch configuration.
 * @param[in] args Array of function arguments, packed in a target specific way.
 * @param[in] arg_sizes Array of function argument sizes.
 * @param[in] num_args Number of arguments in the args and arg_sizes arrays.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args);

typedef M_driver_status_t (*M_driver_queue_execute_func_ptr_t)(
    M_driver_queue_t queue, M_driver_function_t function,
    M_driver_queue_execute_config_t *config, void **args, uint64_t *arg_sizes,
    uint32_t num_args);

/**
 * Record an event on the queue.
 *
 * The event will be signaled once all operations enqueued on the queue before
 * this call have completed.
 *
 * @param queue (in) The queue to record the event on.
 * @param event (in) The event to record.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event);

typedef M_driver_status_t (*M_driver_queue_record_event_func_ptr_t)(
    M_driver_queue_t queue, M_driver_event_t event);

/**
 * Make the queue wait until an event has been signaled.
 *
 * All operations enqueued on the queue after this call will not begin
 * execution until the event has completed.
 *
 * @param queue (in) The queue that should wait.
 * @param event (in) The event to wait on.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
                                                 M_driver_event_t *events,
                                                 uint32_t num_events);

typedef M_driver_status_t (*M_driver_queue_wait_for_events_func_ptr_t)(
    M_driver_queue_t queue, M_driver_event_t *events, uint32_t num_events);

/**
 * Block the calling thread until all operations on the queue have completed.
 *
 * @param[in] queue The queue to synchronize on.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue);

typedef M_driver_status_t (*M_driver_queue_synchronize_func_ptr_t)(
    M_driver_queue_t queue);

/**
 * Query whether the queue is implemented as an in-order stream under the hood.
 *
 * @param[in] queue The queue to inspect.
 * @param[out] is_stream Set to true if the queue is natively in-order.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream);

typedef M_driver_status_t (*M_driver_queue_is_stream_func_ptr_t)(
    M_driver_queue_t queue, bool *is_stream);

/**
 * Common queue properties
 */
#define M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE "native_handle"

/**
 * Retrieve a queue property.
 *
 * @param[in] queue The queue to inspect.
 * @param[in] property_name The name of the property as a null-terminated
 * string.
 * @param[out] value Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * - e.g. "native_handle" to retrieve the backend stream/queue object as a
 *   uint64 (a raw handle value, not an assigned id; used as the DLPack
 *   ``__dlpack__(stream=...)`` handle).
 * @return Standard Status Protocol,
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid,
 *   or M_DRIVER_STATUS_UNIMPLEMENTED if the plugin exposes no native handle.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value);

typedef M_driver_status_t (*M_driver_queue_property_func_ptr_t)(
    M_driver_queue_t queue, const char *property_name, void *value);

/**
 * Enqueue a host function callback on the queue.
 *
 * The callback runs on a driver thread once all preceding work on the queue
 * has completed, and receives `user_data` as its only argument. The callback
 * must not call any device APIs.
 *
 * @param[in] queue The queue to enqueue the callback on.
 * @param[in] fn The callback function.
 * @param[in] user_data Opaque pointer passed through to `fn` when it runs.
 * @return Standard Status Protocol. Backends without host-callback support
 *         report a non-success status.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_launch_host_func(M_driver_queue_t queue,
                                                  void (*fn)(void *),
                                                  void *user_data);

typedef M_driver_status_t (*M_driver_queue_launch_host_func_func_ptr_t)(
    M_driver_queue_t queue, void (*fn)(void *), void *user_data);

/**
 * Enqueue a wait until the 64-bit value at `device_address` equals `value`.
 *
 * The queue stalls at this node until the device-visible 64-bit slot holds
 * `value`; preceding work is unaffected and subsequent work does not start
 * until the wait is satisfied.
 *
 * @param[in] queue The queue to enqueue the wait on.
 * @param[in] device_address Device-visible address of the 64-bit slot.
 * @param[in] value The value to wait for (equality).
 * @return Standard Status Protocol. Backends without stream memory ops
 *         report a non-success status.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_wait_value64(M_driver_queue_t queue,
                                              uint64_t device_address,
                                              uint64_t value);

typedef M_driver_status_t (*M_driver_queue_wait_value64_func_ptr_t)(
    M_driver_queue_t queue, uint64_t device_address, uint64_t value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_QUEUE_H
