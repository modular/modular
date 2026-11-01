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
#ifndef M_DRIVER_DEVICE_H
#define M_DRIVER_DEVICE_H

// This file must not be included directly. #include "M_driver_api.h" instead.

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Retrieve the number of devices supported by this driver implementation.
 *
 * @param[out] count The number of devices supported by the driver. Must be
 * greater than or equal to 0. In case of error, the pointed-to value must
 * remain unchanged.
 * @return The status code.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count);

typedef M_driver_status_t (*M_driver_device_count_func_ptr_t)(
    M_driver_driver_t driver, int64_t *count);

/**
 * Retrieve a device by its ID.
 *
 * @param[in] device_id The ID of the device to retrieve. Must be in the range
 * [0, `count`-1] where `count` is the value previously returned by
 * M_driver_device_count().
 * @param[out] device The device.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_DEVICE if device_id
           is out of range or if the device_id is within range but the device
           cannot be accessed for any reason.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                      int64_t device_id,
                                      M_driver_device_t *device);

typedef M_driver_status_t (*M_driver_device_get_func_ptr_t)(
    M_driver_driver_t driver, int64_t device_id, M_driver_device_t *device);

/**
 * Property names for `M_driver_device_property`.
 *
 * - "dlpack_device" / "dlpack_device_pinned": write an
 *   `M_driver_dlpack_device`; the pinned variant reports the device's
 *   host-pinned DLPack device type.
 * - "name": null-terminated device name string. The value parameter is a
 *   `char *` buffer; the plugin writes up to 256 bytes including the
 *   terminator.
 * - "host_unified": write an `int64_t`, 1 when the device and the host draw
 *   from one physical memory pool. Hardware topology, not addressability.
 * - "max_allocation_size": write a `uint64_t`, the largest single allocation
 *   in bytes. Optional; without it, callers assume the context's total memory.
 */
#define M_DRIVER_DEVICE_PROPERTY_MAX_LEN 256
#define M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE "dlpack_device"
#define M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED "dlpack_device_pinned"
#define M_DRIVER_DEVICE_PROPERTY_NAME "name"
#define M_DRIVER_DEVICE_PROPERTY_ARCH "arch"
#define M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED "host_unified"
#define M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE "max_allocation_size"

/**
 * Retrieve an integer-valued property of a device, as 64-bit integer.
 *
 * @param[in] device The device to retrieve the property from.
 * @param[in] property_name The name of the property to retrieve.
 * @param[out] value Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * @return Standard Status Protocol or one of
 * - M_DRIVER_STATUS_BAD_DEVICE if the device is not valid.
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value);

typedef M_driver_status_t (*M_driver_device_property_func_ptr_t)(
    M_driver_device_t device, const char *property_name, void *value);

/**
 * Retrieve a numeric device attribute by its integer ID.
 *
 * Attribute IDs follow a backend-defined enumeration; a plugin whose native
 * enumeration differs translates the ID to its equivalent.
 *
 * @param[in] device The device to query.
 * @param[in] attribute The attribute ID.
 * @param[out] value The integer attribute value.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_DEVICE.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_attribute(M_driver_device_t device,
                                            int32_t attribute, int32_t *value);

typedef M_driver_status_t (*M_driver_device_attribute_func_ptr_t)(
    M_driver_device_t device, int32_t attribute, int32_t *value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_DEVICE_H
