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

#ifndef M_DRIVER_FUNCTION_H
#define M_DRIVER_FUNCTION_H

#include "M_driver_bundle.h"
#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a named function from a previously registered runtime code bundle.
 *
 * @param[in] context The context to load the function into.
 * @param[in] bundle A runtime code bundle obtained from
 *                   M_driver_bundle_load().
 * @param[in] function_name Null-terminated name of the function to retrieve.
 * @param[in] function_name_len Length of function_name in bytes.
 * @param[out] function The loaded function handle.
 * @return M_DRIVER_STATUS_SUCCESS, or:
 * - M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
 * - M_DRIVER_STATUS_BAD_BUNDLE if the bundle ptr is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function);

typedef M_driver_status_t (*M_driver_function_load_func_ptr_t)(
    M_driver_context_t context, M_driver_runtime_bundle_t bundle,
    const char *function_name, uint64_t function_name_len,
    M_driver_function_t *function);

/**
 * Unload a previously loaded function and release associated resources.
 *
 * @param[in] context The context the function was loaded into.
 * @param[in] function The function to unload.
 * @return M_DRIVER_STATUS_SUCCESS, or:
 * - M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
 * - M_DRIVER_STATUS_BAD_FUNCTION if the function ptr is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function);

typedef M_driver_status_t (*M_driver_function_unload_func_ptr_t)(
    M_driver_context_t context, M_driver_function_t function);

/**
 * Common function properties
 */
#define M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE "native_handle"

/**
 * Retrieve a function property.
 *
 * @param[in] context The context the function was loaded into.
 * @param[in] function The function to inspect.
 * @param[in] property_name The name of the property as a null-terminated
 * string.
 * @param[out] value Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * - e.g. "native_handle" to retrieve the native driver-level module handle
 *   (the module the function was loaded from) as a uint64 (a raw handle
 *   value, not an assigned id; used for vendor-library interop).
 * @return Standard Status Protocol,
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid
 *   or the plugin does not expose it.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_property(M_driver_context_t context,
                                             M_driver_function_t function,
                                             const char *property_name,
                                             void *value);

typedef M_driver_status_t (*M_driver_function_property_func_ptr_t)(
    M_driver_context_t context, M_driver_function_t function,
    const char *property_name, void *value);

/**
 * Set a backend function attribute on a loaded function.
 *
 * `attribute` and `value` mirror the driver's function-attribute enum; the
 * codes match those enum values. The primary use is raising the dynamic
 * shared-memory cap before launching a kernel that needs more than the
 * default.
 *
 * @param[in] context The context the function was loaded into.
 * @param[in] function The function to modify.
 * @param[in] attribute The attribute code to set.
 * @param[in] value The value to assign to the attribute.
 * @return Standard Status Protocol. Backends that do not support setting
 *         function attributes report a non-success status.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_set_attribute(M_driver_context_t context,
                                                  M_driver_function_t function,
                                                  int32_t attribute,
                                                  int32_t value);

typedef M_driver_status_t (*M_driver_function_set_attribute_func_ptr_t)(
    M_driver_context_t context, M_driver_function_t function, int32_t attribute,
    int32_t value);

/**
 * Query the maximum number of blocks of a function that can be resident on a
 * single multiprocessor, for the given block size and dynamic shared-memory
 * usage.
 *
 * @param[in] context The context the function was loaded into.
 * @param[in] function The function to query.
 * @param[in] block_size Number of threads per block.
 * @param[in] dynamic_shared_memory_bytes Dynamic shared memory per block.
 * @param[out] num_blocks The max number of resident blocks per multiprocessor.
 * @return Standard Status Protocol. Backends that do not support occupancy
 *         queries report a non-success status.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    M_driver_context_t context, M_driver_function_t function,
    int32_t block_size, uint64_t dynamic_shared_memory_bytes,
    int32_t *num_blocks);

typedef M_driver_status_t (
    *M_driver_function_occupancy_max_active_blocks_func_ptr_t)(
    M_driver_context_t context, M_driver_function_t function,
    int32_t block_size, uint64_t dynamic_shared_memory_bytes,
    int32_t *num_blocks);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_FUNCTION_H
