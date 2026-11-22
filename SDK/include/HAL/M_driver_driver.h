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
#ifndef M_DRIVER_DRIVER_H
#define M_DRIVER_DRIVER_H

// This file must not be included directly. #include "M_driver_api.h" instead.

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform initialization activities prior to using other functions in this API.
 *
 * @param[in,out] version On entry, the version of the driver interface that
 * Modular was built with. Before it returns, whatever the status, the plugin
 * MUST overwrite it with the version it implements. `M_driver_version`
 * describes how Modular decides whether the two versions are compatible.
 * @param[out] driver The driver object, which may store any persistent state.
 * Modular sets `*driver` to NULL before the call.
 *
 * @return M_DRIVER_STATUS_SUCCESS if the driver was created and is usable. A
 *         driver without devices is usable: the plugin returns
 *         M_DRIVER_STATUS_SUCCESS and `M_driver_device_count()` reports 0.
 *         M_DRIVER_STATUS_INCOMPATIBLE_VERSION if the plugin does not implement
 *         Modular's major version. When only the minor versions differ, the
 *         plugin must not return this status; Modular compares the versions.
 *         Any other status if the plugin cannot create a usable driver. This
 *         includes M_DRIVER_STATUS_NO_DEVICES, which a plugin returns when it
 *         cannot operate without a device.
 *
 * When it does not return M_DRIVER_STATUS_SUCCESS, the plugin either leaves
 * `*driver` NULL or writes a handle that exists only to describe the failure.
 * Modular may pass that handle and the returned status to
 * `M_driver_status_message()`, then destroys the handle with
 * `M_driver_destroy()`, and calls no other function with it. Without a handle,
 * Modular reports the status code alone.
 *
 * When the call succeeds but Modular rejects the version the plugin wrote,
 * Modular destroys the driver with `M_driver_destroy()`.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *driver);

typedef M_driver_status_t (*M_driver_create_func_ptr_t)(
    M_driver_version *version, M_driver_driver_t *driver);

/**
 * Perform shutdown activities after using all other functions in this API,
 * except for M_driver_status_message().
 *
 * @param[in] driver The driver to destroy.
 *
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_destroy(M_driver_driver_t driver);

typedef M_driver_status_t (*M_driver_destroy_func_ptr_t)(
    M_driver_driver_t driver);

/**
 * Common driver properties
 */
#define M_DRIVER_PROPERTY_API "api"
#define M_DRIVER_PROPERTY_VERSION "version"

/**
 * Retrieve an integer-valued property.
 *
 * @param[in] property_name The name of the property as a null-terminated
 * string.
 * @param[out] value Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * @return Standard Status Protocol or
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_property(M_driver_driver_t handle,
                                    const char *property_name, void *value);

typedef M_driver_status_t (*M_driver_property_func_ptr_t)(
    M_driver_driver_t handle, const char *property_name, void *value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_DRIVER_H
