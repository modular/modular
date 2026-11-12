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

#ifndef M_DRIVER_FLAG_H
#define M_DRIVER_FLAG_H

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A completion flag is a 64-bit counter the host writes and a queue waits on.
 * CUDA and HIP back it with a pinned host slot the device polls, Metal with an
 * `MTLSharedEvent`.
 *
 * The extension is optional. A plugin exports every symbol of a capability
 * group or none of them; a caller treats a missing group as absent, and the
 * plugin still loads:
 *
 * - CompletionFlagV1: `M_driver_flag_create`, `M_driver_flag_destroy`,
 *   `M_driver_flag_signal`, `M_driver_flag_load`, `M_driver_flag_conditions`
 *   and `M_driver_queue_wait_flag`.
 * - CompletionFlagResetV1: `M_driver_flag_reset`. Requires CompletionFlagV1.
 * - CompletionFlagAddressV1: `M_driver_flag_device_address`. Requires
 *   CompletionFlagV1.
 *
 * Values never decrease between resets. Keep values below 2^63: HIP compares
 * them as signed.
 *
 * `M_driver_flag_signal` is a release; `M_driver_flag_load` and a satisfied
 * queue wait are acquires.
 *
 * A queue wait stays pending until the queue has executed past it, for example
 * after `M_driver_queue_synchronize` or once an event recorded after the wait
 * completes, even if the flag already holds a value that satisfies it.
 *
 * Destroying a flag while a wait on it is pending is undefined behavior. On
 * CUDA and HIP, destroy frees pinned memory, which synchronizes the device, so
 * destroying any flag while a wait on any flag on that device is pending is
 * undefined behavior. On Metal it is memory-safe: a queued wait holds its own
 * reference to the event. A wait that the flag did not satisfy before destroy
 * blocks the queue forever.
 *
 * A null `M_driver_flag_t` argument returns M_DRIVER_STATUS_INVALID_ARG.
 */
typedef struct M_driver_flag *M_driver_flag_t;

/**
 * Wait conditions, each read as `flag value <op> target`. Each plugin reports
 * a fixed set through `M_driver_flag_conditions`; every plugin that exports
 * CompletionFlagV1 supports GEQ.
 */
typedef enum {
  M_DRIVER_FLAG_CONDITION_GEQ = 1 << 0,
  M_DRIVER_FLAG_CONDITION_EQ = 1 << 1,
  M_DRIVER_FLAG_CONDITION_GT = 1 << 2,
  M_DRIVER_FLAG_CONDITION_LT = 1 << 3,
  M_DRIVER_FLAG_CONDITION_LEQ = 1 << 4,
} M_driver_flag_condition_t;

/**
 * CompletionFlagV1: create a completion flag whose value is 0.
 *
 * @param[in] context The context that owns the flag.
 * @param[out] out_flag The new flag.
 * @return Standard Status Protocol, M_DRIVER_STATUS_BAD_CONTEXT if `context`
 *         is not valid, M_DRIVER_STATUS_INVALID_ARG if `out_flag` is null, or
 *         M_DRIVER_STATUS_OUT_OF_MEMORY.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_create(M_driver_context_t context,
                                       M_driver_flag_t *out_flag);

typedef M_driver_status_t (*M_driver_flag_create_func_ptr_t)(
    M_driver_context_t context, M_driver_flag_t *out_flag);

/**
 * Destroy a completion flag. See the pending-wait rule above.
 *
 * @param[in] context The context that created the flag.
 * @param[in] flag The flag to destroy.
 * @return Standard Status Protocol, M_DRIVER_STATUS_BAD_CONTEXT if `context`
 *         is not valid, or M_DRIVER_STATUS_INVALID_ARG if `flag` is null.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_destroy(M_driver_context_t context,
                                        M_driver_flag_t flag);

typedef M_driver_status_t (*M_driver_flag_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_flag_t flag);

/**
 * Set the flag's value from the host (a release store), releasing queue waits
 * the new value satisfies.
 *
 * `value` must not be lower than one signaled since creation or the last
 * reset. A plugin whose flag cannot decrease returns
 * M_DRIVER_STATUS_INVALID_ARG for a lower value; on others it is undefined.
 *
 * @param[in] flag The flag to signal.
 * @param[in] value The new value.
 * @return Standard Status Protocol or M_DRIVER_STATUS_INVALID_ARG.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_signal(M_driver_flag_t flag, uint64_t value);

typedef M_driver_status_t (*M_driver_flag_signal_func_ptr_t)(
    M_driver_flag_t flag, uint64_t value);

/**
 * Read the flag's value from the host (an acquire load).
 *
 * @param[in] flag The flag to read.
 * @param[out] out_value The flag's value.
 * @return Standard Status Protocol or M_DRIVER_STATUS_INVALID_ARG if `flag` or
 *         `out_value` is null.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_load(M_driver_flag_t flag, uint64_t *out_value);

typedef M_driver_status_t (*M_driver_flag_load_func_ptr_t)(M_driver_flag_t flag,
                                                           uint64_t *out_value);

/**
 * Report the wait conditions `device` supports, as a bitwise OR of
 * `M_driver_flag_condition_t` values. The set is fixed per plugin and always
 * includes M_DRIVER_FLAG_CONDITION_GEQ.
 *
 * @param[in] device The device whose queues would wait.
 * @param[out] out_mask The supported conditions.
 * @return Standard Status Protocol, M_DRIVER_STATUS_BAD_DEVICE if `device` is
 *         not valid, or M_DRIVER_STATUS_INVALID_ARG if `out_mask` is null.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_conditions(M_driver_device_t device,
                                           uint32_t *out_mask);

typedef M_driver_status_t (*M_driver_flag_conditions_func_ptr_t)(
    M_driver_device_t device, uint32_t *out_mask);

/**
 * Make work enqueued on `queue` after this call wait until the flag's value
 * satisfies `condition` against `value`. The queue evaluates the wait; this
 * call does not block the host.
 *
 * While the wait is pending, the thread that will signal the flag must not
 * make a call that waits for the queue or the device: synchronize, a copy to
 * or from pageable host memory (small host-to-device copies may return on
 * CUDA and HIP), freeing pinned or device memory, or destroying a flag. On
 * Metal, memset and fill also wait.
 *
 * @param[in] queue The queue that waits.
 * @param[in] flag The flag to test.
 * @param[in] value The target the condition compares against. Keep it below
 *            2^63: HIP compares it as signed.
 * @param[in] condition A single `M_driver_flag_condition_t` value.
 * @return Standard Status Protocol, M_DRIVER_STATUS_BAD_QUEUE if `queue` is
 *         not valid, M_DRIVER_STATUS_INVALID_ARG if `flag` is null, or
 *         M_DRIVER_STATUS_UNIMPLEMENTED if `condition` is not in the mask
 *         `M_driver_flag_conditions` reports for the queue's device.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_wait_flag(M_driver_queue_t queue,
                                           M_driver_flag_t flag, uint64_t value,
                                           M_driver_flag_condition_t condition);

typedef M_driver_status_t (*M_driver_queue_wait_flag_func_ptr_t)(
    M_driver_queue_t queue, M_driver_flag_t flag, uint64_t value,
    M_driver_flag_condition_t condition);

/**
 * CompletionFlagResetV1 (optional): set the value back to 0 so later signals
 * may start low again. The flag keeps its identity and, with
 * CompletionFlagAddressV1, its device address.
 *
 * Resetting while a wait on the flag is pending, as defined above, is
 * undefined behavior.
 *
 * @param[in] flag The flag to reset.
 * @return Standard Status Protocol, M_DRIVER_STATUS_INVALID_ARG if `flag` is
 *         null, or M_DRIVER_STATUS_OUT_OF_MEMORY.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_reset(M_driver_flag_t flag);

typedef M_driver_status_t (*M_driver_flag_reset_func_ptr_t)(
    M_driver_flag_t flag);

/**
 * CompletionFlagAddressV1 (optional): report the device address of the flag's
 * 64-bit value, for kernels or vendor APIs that access it directly. Stable for
 * the flag's lifetime.
 *
 * @param[in] flag The flag to query.
 * @param[out] out_address The device address of the flag's value.
 * @return Standard Status Protocol or M_DRIVER_STATUS_INVALID_ARG if `flag` or
 *         `out_address` is null.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_flag_device_address(M_driver_flag_t flag,
                                               uint64_t *out_address);

typedef M_driver_status_t (*M_driver_flag_device_address_func_ptr_t)(
    M_driver_flag_t flag, uint64_t *out_address);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_FLAG_H
