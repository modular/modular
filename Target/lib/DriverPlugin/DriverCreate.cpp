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

#include "Target/DriverCreate.h"

#include <string>

namespace M::Driver::Plugin {

namespace {

std::string formatVersion(const M_driver_version &version) {
  return std::to_string(version.major) + "." + std::to_string(version.minor) +
         "." + std::to_string(version.patch);
}

} // namespace

bool isCompatibleDriverVersion(const M_driver_version &host,
                               const M_driver_version &plugin) {
  return plugin.major == host.major && plugin.minor >= host.minor;
}

ErrorOr<M_driver_driver_t>
createDriver(llvm::StringRef path, M_driver_create_func_ptr_t create,
             M_driver_destroy_func_ptr_t destroy,
             M_driver_status_message_func_ptr_t statusMessage) {
  M_driver_version version = hostDriverVersion;
  M_driver_driver_t driver = nullptr;
  M_driver_status_t status = create(&version, &driver);

  if (status == M_DRIVER_STATUS_INCOMPATIBLE_VERSION ||
      (status == M_DRIVER_STATUS_SUCCESS &&
       !isCompatibleDriverVersion(hostDriverVersion, version))) {
    if (driver)
      destroy(driver);
    // A plugin built before the header required the write-back refuses
    // without one, which would otherwise read as the host's own version.
    if (status == M_DRIVER_STATUS_INCOMPATIBLE_VERSION &&
        version.major == hostDriverVersion.major &&
        version.minor == hostDriverVersion.minor &&
        version.patch == hostDriverVersion.patch)
      return Error("driver plugin " + path.str() +
                   " rejected this build's driver interface " +
                   formatVersion(hostDriverVersion) +
                   " without reporting the version it implements");
    return Error("driver plugin " + path.str() +
                 " implements driver interface " + formatVersion(version) +
                 ", which is incompatible with this build's " +
                 formatVersion(hostDriverVersion) +
                 ": the major versions must match and the plugin's minor "
                 "version must be at least this build's");
  }

  if (status != M_DRIVER_STATUS_SUCCESS) {
    std::string detail = "the plugin returned no handle to describe it";
    if (driver) {
      char message[1024] = {};
      detail = statusMessage(driver, status, message, sizeof(message)) ==
                       M_DRIVER_STATUS_SUCCESS
                   ? std::string(message)
                   : "the plugin's handle gave no message";
      destroy(driver);
    }
    return Error("driver plugin " + path.str() +
                 " failed to create its driver (status " +
                 std::to_string(status) + "): " + detail);
  }

  if (!driver)
    return Error("driver plugin " + path.str() +
                 " reported success from M_driver_create without a driver");
  return driver;
}

} // namespace M::Driver::Plugin
