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
// A plugin that exports exactly the entry points `Plugin::loadPlugins`
// requires, so `DriverPluginLoadTest` can load one without a vendor runtime. It
// has no devices.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_api.h"

#include <cstdint>
#include <cstring>

namespace {
int driverHandle;
} // namespace

extern "C" {

M_driver_status_t M_driver_status_message(M_driver_driver_t, M_driver_status_t,
                                          char *messageBuffer,
                                          int64_t messageBufferSize) {
  if (messageBufferSize > 0)
    messageBuffer[0] = '\0';
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_create(M_driver_version *,
                                  M_driver_driver_t *driver) {
  *driver = reinterpret_cast<M_driver_driver_t>(&driverHandle);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t) {
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_property(M_driver_driver_t, const char *propertyName,
                                    void *value) {
  if (std::strcmp(propertyName, M_DRIVER_PROPERTY_API) != 0)
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  *static_cast<const char **>(value) = "legacystub";
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_count(M_driver_driver_t, int64_t *count) {
  *count = 0;
  return M_DRIVER_STATUS_SUCCESS;
}

M_DRIVER_API_EXPORT M_driver_status_t M_driver_device_get_native(int64_t,
                                                                 void **,
                                                                 int64_t) {
  return M_DRIVER_STATUS_BAD_DEVICE;
}

M_DRIVER_API_EXPORT M_driver_status_t
M_driver_device_arch_name_primary(const char **) {
  return M_DRIVER_STATUS_BAD_DEVICE;
}

M_DRIVER_API_EXPORT M_driver_status_t
M_driver_create_compilation_device(const char *, void **) {
  return M_DRIVER_STATUS_BAD_DEVICE;
}

M_DRIVER_API_EXPORT M_driver_status_t
M_driver_destroy_compilation_device(void *) {
  return M_DRIVER_STATUS_SUCCESS;
}

} // extern "C"
