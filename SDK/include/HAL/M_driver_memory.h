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

#ifndef M_DRIVER_MEMORY_H
#define M_DRIVER_MEMORY_H

#include <stdbool.h>

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate pinned (page-locked) host memory for the given context.
 *
 * @param[in] context The context to create the memory for.
 * @param[in] size Size in bytes of the allocation.
 * @param[out] memory Pointer to the address of the newly allocated memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_CONTEXT if the context is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_alloc_pinned_func_ptr_t)(
    M_driver_context_t context, uint64_t size, M_driver_memory_t *memory);

/**
 * Free pinned (page-locked) host memory for the given context.
 *
 * @param[in] context The context to free the memory from.
 * @param[in] memory The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory);

typedef M_driver_status_t (*M_driver_memory_free_pinned_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_t memory);

/**
 * Allocate memory for the given context.
 *
 * @param[in] context The context to create the memory for.
 * @param[in] size Size in bytes of the allocation.
 * @param[out] memory Pointer to the address of the newly allocated memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_CONTEXT if the context is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
                                             uint64_t size,
                                             M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_alloc_sync_func_ptr_t)(
    M_driver_context_t context, uint64_t size, M_driver_memory_t *memory);

/**
 * Free memory for the given context.
 *
 * @param[in] context The context to free the memory from.
 * @param[in] memory The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory);

typedef M_driver_status_t (*M_driver_memory_free_sync_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_t memory);

/**
 * Wrap an existing memory region in a memory handle, without allocating.
 *
 * When @p owning is false the region is treated as externally owned
 * (e.g. another library's allocation on the same device): the plugin
 * never frees it, freeing the handle releases only the plugin's
 * bookkeeping, and the caller must keep the region alive for the
 * handle's lifetime. When @p owning is true, freeing the handle frees
 * the region through the plugin's normal path — this is only valid for
 * an address that originated from this plugin's own allocator (e.g. one
 * previously released with M_driver_memory_unwrap). Only device
 * allocations round-trip this way: a wrapped handle always frees through
 * the device path, never the pinned-host path.
 *
 * @param[in] context The context the region belongs to.
 * @param[in] address Base address of the region, in the address space the
 *            plugin uses for device memory. Must be non-zero; a zero
 *            address is rejected with M_DRIVER_STATUS_INVALID_ARG.
 * @param[in] size Size in bytes of the region.
 * @param[in] owning Whether freeing the handle should free the region.
 * @param[out] memory Pointer to the handle wrapping the region.
 * @return Standard Status Protocol, M_DRIVER_STATUS_UNIMPLEMENTED if the
 *         plugin cannot represent wrapped allocations, or
 *         M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_wrap(M_driver_context_t context,
                                       uint64_t address, uint64_t size,
                                       bool owning, M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_wrap_func_ptr_t)(
    M_driver_context_t context, uint64_t address, uint64_t size, bool owning,
    M_driver_memory_t *memory);

/**
 * Release ownership of the underlying region from a memory handle and
 * return its address.
 *
 * The inverse of M_driver_memory_wrap: after this call the handle is
 * non-owning — freeing it releases only the plugin's bookkeeping — and
 * the caller becomes responsible for the region at the returned
 * address. The handle itself stays valid until freed. Calling this on
 * an already non-owning handle only returns the address.
 *
 * @param[in] context The context the handle belongs to.
 * @param[in] memory The handle to release ownership from.
 * @param[out] address Base address of the underlying region.
 * @return Standard Status Protocol, M_DRIVER_STATUS_UNIMPLEMENTED if the
 *         plugin cannot represent non-owning handles,
 *         M_DRIVER_STATUS_BAD_MEMORY if the handle is not valid, or
 *         M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_unwrap(M_driver_context_t context,
                                         M_driver_memory_t memory,
                                         uint64_t *address);

typedef M_driver_status_t (*M_driver_memory_unwrap_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_t memory, uint64_t *address);

/**
 * Asynchronously allocate memory on the given queue.
 *
 * @param[in] queue The queue to allocate memory on.
 * @param[in] size Size in bytes of the allocation.
 * @param[out] memory Pointer to the address of the newly allocated memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_QUEUE if the queue is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_alloc_async_func_ptr_t)(
    M_driver_queue_t queue, uint64_t size, M_driver_memory_t *memory);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory);
/**
 * Asynchronously free memory on the given queue.
 *
 * @param[in] queue The queue to free the memory from.
 * @param[in] memory The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_QUEUE if the
 *         queue is not valid.
 */
typedef M_driver_status_t (*M_driver_memory_free_async_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_t memory);

/**
 * Common memory properties
 */
#define M_DRIVER_MEMORY_PROPERTY_ADDRESS "address"

/**
 * Host-dereferenceable address of an allocation. Only meaningful for pinned
 * host memory created with M_driver_memory_alloc_pinned.
 */
#define M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS "host_address"

/**
 * Retrieve a memory property.
 *
 * @param[in] property_name The name of the property as a null-terminated
 * string.
 * @param[out] value Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * - e.g. "address" to retrieve the device memory address
 * @return Standard Status Protocol or
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
                                           const char *property_name,
                                           void *value);

typedef M_driver_status_t (*M_driver_memory_property_func_ptr_t)(
    M_driver_memory_t memory, const char *property_name, void *value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_MEMORY_H
