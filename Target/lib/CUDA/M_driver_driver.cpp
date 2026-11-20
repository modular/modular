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

#include "Driver.h"
#include "HAL/M_driver_api.h"
#include "llvm/ADT/StringRef.h"

M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *driver) {
  if (version == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (driver == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (version->major != M_DRIVER_INTERFACE_VERSION_MAJOR)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;

  version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
  version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
  version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;

  DriverResult<M_driver_driver_t> result = M_driver_driver::create();
  if (!result)
    return result.error();

  *driver = *result;
  return (*driver)->initStatus;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  if (handle == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  return M_driver_driver::destroy(handle);
}

M_driver_status_t M_driver_property([[maybe_unused]] M_driver_driver_t handle,
                                    const char *property_name, void *value) {
  llvm::StringRef name(property_name);
  if (name == M_DRIVER_PROPERTY_API)
    *static_cast<const char **>(value) = "CUDA";
  else if (name == M_DRIVER_PROPERTY_VERSION) {
    M_driver_version *version = static_cast<M_driver_version *>(value);
    version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
    version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
  } else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
