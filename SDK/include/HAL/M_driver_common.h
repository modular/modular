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
#ifndef M_DRIVER_COMMON_H
#define M_DRIVER_COMMON_H

#ifdef __cplusplus
#include <cstring>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define M_DRIVER_API_EXPORT __attribute__((visibility("default")))
#define M_DRIVER_API_LOCAL __attribute__((visibility("hidden")))

/**
 * The major, minor, and patch version of the driver interface.
 * Semantic versioning: https://semver.org/
 *
 * The Modular stack and the hardware plugins may be developed by different
 * organizations at different times. Both Modular and plugins must declare to
 * each other the hardware interface version they implement at plugin
 * initialization, when Modular calls `M_driver_create()`. Plugins must also
 * return their version via the `M_driver_property()` query with the "version"
 * property name.
 *
 * When comparing versions between the Modular stack and the HW plugins,
 * MAJOR version: When same, indicates compatibility.
 *                When different, indicates incompatibility.
 * MINOR version: Relevant only when the major version is the same. Then:
 *                - the same minor version indicates no new functionality and
 *                  that it is safe to proceed.
 *                - When the plugin's minor version is greater than the
 *                  Modular stack's minor version, it is safe to proceed,
 *                  since the plugin is backward compatible.
 *                - When the plugin's minor version is less than Modular's,
 *                  the plugin must overwrite Modular's version with its own
 *                  when returning from `M_driver_create()`. Modular will then
 *                  decide how to proceed.
 * PATCH version: Increases monotonically for each backward compatible bug
 *                fix.
 */
typedef struct M_driver_version {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} M_driver_version;

/* For implementers of this interface */
static uint32_t const M_DRIVER_INTERFACE_VERSION_MAJOR = 0u;
static uint32_t const M_DRIVER_INTERFACE_VERSION_MINOR = 1u;
static uint32_t const M_DRIVER_INTERFACE_VERSION_PATCH = 0u;

/**
 * Common type definitions.
 */
typedef int64_t M_driver_status_t;
typedef struct M_driver_driver *M_driver_driver_t;
typedef struct M_driver_device *M_driver_device_t;
typedef struct M_driver_context *M_driver_context_t;
typedef struct M_driver_memory *M_driver_memory_t;
typedef struct M_driver_queue *M_driver_queue_t;
typedef struct M_driver_event *M_driver_event_t;
typedef struct M_driver_function *M_driver_function_t;

/**
 * An opaque backend-defined handle exposed for vendor-library interop.
 */
typedef intptr_t M_driver_native_handle_t;

/**
 * A view over a device memory allocation.
 *
 * Bundles the owning allocation handle with a byte offset and byte length,
 * describing the span a queue copy/set op should act on. Passed by pointer to
 * those ops. A view is non-owning: it does not free `memory`.
 *
 */
struct M_driver_memory_view {
  M_driver_memory_t memory; ///< The owning allocation handle.
  uint64_t offset;          ///< Byte offset of the view into the allocation.
  uint64_t size;            ///< Byte length of the view.
};

typedef struct M_driver_memory_view *M_driver_memory_view_t;

/**
 * A DLPack device: the `(device_type, device_id)` pair identifying where a
 * tensor lives, per the DLPack protocol. `device_type` holds a DLPack
 * `DLDeviceType` code (e.g. `kDLCUDA`, `kDLCUDAHost`, `kDLMetal`); a plugin
 * fills it from dlpack.h. For host-pinned or CPU memory, `device_id` is 0 (per
 * the DLPack spec). Written by the `M_driver_device_property` "dlpack_device" /
 * "dlpack_device_pinned" queries.
 */
typedef struct M_driver_dlpack_device {
  int32_t device_type;
  int32_t device_id;
} M_driver_dlpack_device;

/**
 * Returned status-codes from all API functions.
 *
 * Implementations are encouraged to use the predefined set below where
 * possible.
 *
 * Implementations may also extend this set by defining additional (custom)
 * status codes, but must use integers less than or equal to
 * M_DRIVER_STATUS_CUSTOM_CODES_BASE for errors and strictly greater than 0
 * for warnings and information.
 *
 * Every status code must generate a human-readable message when supplied to
 * `M_driver_status_message()` (defined below).
 *
 * Custom status codes may be ephemeral.
 * See documentation for M_driver_status_message() for more details.
 */
#define M_DRIVER_STATUS_SUCCESS ((M_driver_status_t)(0))
#define M_DRIVER_STATUS_INCOMPATIBLE_VERSION ((M_driver_status_t)(-1))
#define M_DRIVER_STATUS_UNKNOWN_ERROR ((M_driver_status_t)(-2))
#define M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE ((M_driver_status_t)(-3))
#define M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME ((M_driver_status_t)(-4))
#define M_DRIVER_STATUS_NO_DEVICES ((M_driver_status_t)(-5))
#define M_DRIVER_STATUS_BAD_DRIVER ((M_driver_status_t)(-6))
#define M_DRIVER_STATUS_BAD_DEVICE ((M_driver_status_t)(-7))
#define M_DRIVER_STATUS_BAD_CONTEXT ((M_driver_status_t)(-8))
#define M_DRIVER_STATUS_BAD_MEMORY ((M_driver_status_t)(-9))
#define M_DRIVER_STATUS_BAD_QUEUE ((M_driver_status_t)(-10))
#define M_DRIVER_STATUS_BAD_EVENT ((M_driver_status_t)(-11))
#define M_DRIVER_STATUS_BAD_FUNCTION ((M_driver_status_t)(-12))
#define M_DRIVER_STATUS_OUT_OF_MEMORY ((M_driver_status_t)(-13))
#define M_DRIVER_STATUS_INVALID_ARG ((M_driver_status_t)(-14))
#define M_DRIVER_STATUS_INCOMPATIBLE ((M_driver_status_t)(-15))
#define M_DRIVER_STATUS_BAD_BUNDLE ((M_driver_status_t)(-16))
#define M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED ((M_driver_status_t)(-17))
#define M_DRIVER_STATUS_UNKNOWN_SYMBOL ((M_driver_status_t)(-18))
/* The backend or device cannot do what was asked. */
#define M_DRIVER_STATUS_UNSUPPORTED ((M_driver_status_t)(-19))
/* The plugin has not implemented this entry point or case yet. */
#define M_DRIVER_STATUS_UNIMPLEMENTED ((M_driver_status_t)(-20))
/* Future standard status codes may be added here. */
#define M_DRIVER_NUM_STANDARD_STATUS_CODES (21)

#define M_DRIVER_STATUS_CUSTOM_CODES_BASE ((M_driver_status_t)(-65536))

/**
 * Retrieve human-readable status message corresponding to a status code.
 *
 * @param[in] status A status code.
 * @param[in,out] message_buffer A buffer to write the corresponding status
 *                message into as a null-terminated string.
 * @param[in] message_buffer_size The size of the buffer.
 * @note If the status message is longer than the buffer, the message
 *       must be truncated to fit into the buffer. In any case, the last
 *       character must be the null terminator, '\0'.
 * @note Implementations may support ephemeral status messages: strings bound
 *       to a custom status code till such time M_driver_status_message() is
 *       called. At that time, the plugin may discard the string after copying
 *       it into the buffer and recycle the status code.  Subsequent calls to
 *       M_driver_status_message() for the same status code will return
 *       M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE unless the status code has been
 *       reassigned to a new ephemeral message. Such behavior can be useful
 *       with dynamically-constructed messages that may contain
 *       context-specific information.
 * @return Standard Status Protocol, or
 *         M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE if `status` does not match any
 *           of the above common status codes or other persistent ones that the
 *           implementation maintains, or
 *         M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE if `message_buffer_size` is
 *           less than or equal to 0, or
 *         M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE if `status` is an expired
 *           (ephemeral) custom status code.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_status_message(M_driver_driver_t driver,
                                          M_driver_status_t status,
                                          char *message_buffer,
                                          int64_t message_buffer_size);

typedef M_driver_status_t (*M_driver_status_message_func_ptr_t)(
    M_driver_driver_t driver, M_driver_status_t status, char *message_buffer,
    int64_t message_buffer_size);

/// A borrowed span of bytes.
/// Encoding of the underlying bytes must be explicitly documented
/// at each use-site.
///
/// This does not own the backing allocation, and no M_driver_slice
/// may be stored non-transiently unless additional guarantees are
/// provided by the passing context.
struct M_driver_slice {
  const char *data;
  uint64_t size;
};

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_COMMON_H
