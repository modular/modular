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

#ifndef M_DRIVER_EVENT_H
#define M_DRIVER_EVENT_H

#include "M_driver_common.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Bitmask describing capabilities an event must support.
 */
typedef uint32_t M_driver_event_flags_t;

// Default (no bits set) is intra-GPU synchronization only.
#define M_DRIVER_EVENT_FLAG_NONE ((M_driver_event_flags_t)0u)
// Host may call M_driver_event_synchronize / M_driver_is_event_ready on
// this event.
#define M_DRIVER_EVENT_FLAG_CPU_VISIBLE ((M_driver_event_flags_t)(1u << 0))

// Union of all defined flag bits. Plugins use this to reject unknown bits.
// Update when adding a new flag.
#define M_DRIVER_EVENT_FLAGS_ALL (M_DRIVER_EVENT_FLAG_CPU_VISIBLE)

/**
 * Create an event for the given context.
 *
 * @param[in] context The context to create the event for.
 * @param[in] flags Bitmask describing required capabilities.
 * @param[out] event The event.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_flags_t flags,
                                        M_driver_event_t *event);

typedef M_driver_status_t (*M_driver_event_create_func_ptr_t)(
    M_driver_context_t context, M_driver_event_flags_t flags,
    M_driver_event_t *event);

/**
 * Destroy an event.
 *
 * @param[in] context The context the event belongs to.
 * @param[in] event The event to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event);

typedef M_driver_status_t (*M_driver_event_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event);

/**
 * Block the calling thread until the event has completed.
 *
 * @param[in] context The context the event belongs to.
 * @param[in] event The event to synchronize on.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event);

typedef M_driver_status_t (*M_driver_event_synchronize_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event);

/**
 * Query whether an event has completed without blocking.
 *
 * @param[in] context The context the event belongs to.
 * @param[in] event The event to query.
 * @param[out] is_ready Set to true if the event has completed, false
 *             otherwise.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready);

typedef M_driver_status_t (*M_driver_is_event_ready_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event, bool *is_ready);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_EVENT_H
