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
// has no devices, unless `LEGACY_STUB_PLUGIN_DEVICE_COUNT` names a count: it
// then claims that many without being able to open one, which is what lets a
// test observe whether the plugin reached the registry the device lookup reads.
//
// Like a vendor runtime, it refuses a second live driver in one process, so two
// copies of the loader cannot both register it without one of them reporting
// the failure.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_api.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
int driverHandle;
bool driverLive = false;
} // namespace

extern "C" {

M_driver_status_t M_driver_status_message(M_driver_driver_t,
                                          M_driver_status_t status,
                                          char *messageBuffer,
                                          int64_t messageBufferSize) {
  const char *message = status == M_DRIVER_STATUS_BAD_DRIVER
                            ? "legacystub driver already created"
                            : "";
  if (messageBufferSize > 0) {
    std::strncpy(messageBuffer, message, messageBufferSize - 1);
    messageBuffer[messageBufferSize - 1] = '\0';
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_create(M_driver_version *,
                                  M_driver_driver_t *driver) {
  *driver = nullptr;
  if (driverLive)
    return M_DRIVER_STATUS_BAD_DRIVER;
  driverLive = true;
  *driver = reinterpret_cast<M_driver_driver_t>(&driverHandle);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t) {
  driverLive = false;
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
  const char *configured = std::getenv("LEGACY_STUB_PLUGIN_DEVICE_COUNT");
  *count = configured ? std::atoll(configured) : 0;
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
