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
// This file implements the driver-level entry points of the CPU HAL plugin:
// driver lifecycle, property queries, and status messages. The CPU plugin
// bridges the HAL onto AsyncRT, mirroring the behavior of the legacy
// M::Driver::CpuDeviceContext.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Common.h"
#include "HAL/M_driver_api.h"
#include "llvm/ADT/StringRef.h"

#include <new>

M_driver_driver_t M_driver_driver::create() {
  return new (std::nothrow) M_driver_driver();
}

void M_driver_driver::destroy(M_driver_driver_t driver) { delete driver; }

M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *out_driver) {
  if (version == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (out_driver == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (version->major != M_DRIVER_INTERFACE_VERSION_MAJOR)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;

  version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
  version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
  version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;

  M_driver_driver_t driver = M_driver_driver::create();
  if (!driver)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  *out_driver = driver;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  if (handle == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  M_driver_driver::destroy(handle);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_property([[maybe_unused]] M_driver_driver_t handle,
                                    const char *property_name, void *value) {
  llvm::StringRef name(property_name);
  if (name == M_DRIVER_PROPERTY_API) {
    *static_cast<const char **>(value) = "cpu";
  } else if (name == M_DRIVER_PROPERTY_VERSION) {
    M_driver_version *version = static_cast<M_driver_version *>(value);
    version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
    version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
  } else {
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_status_message(M_driver_driver_t driver,
                                          M_driver_status_t status,
                                          char *message_buffer,
                                          int64_t message_buffer_size) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  return driver->status.getMessage(status, message_buffer, message_buffer_size);
}
