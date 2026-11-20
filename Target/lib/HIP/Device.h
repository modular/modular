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

#ifndef TARGET_LIB_HIP_DEVICE_H
#define TARGET_LIB_HIP_DEVICE_H

#include "Driver.h"
#include "Error.h"
#include "HIPApi.h"
#include "llvm/ADT/Twine.h"

struct M_driver_device {
private:
  M_driver_driver &driver;
  int deviceId;

public:
  DriverResult<void> deviceMethodPreamble() {
    if (hipError_t r = this->driver.hip().hipSetDevice(this->deviceId);
        r != hipSuccess)
      return tl::unexpected(M::Target::HIP::hipErrorStatus(
          &this->driver, r,
          "hipSetDevice device " + llvm::Twine(this->deviceId)));
    return {};
  }
  [[nodiscard]] static DriverResult<M_driver_device_t>
  create(M_driver_driver &drv, int64_t deviceId);

  static M_driver_status_t destroy(M_driver_device_t device);

  explicit M_driver_device(M_driver_driver &driver, int deviceId)
      : driver(driver), deviceId(deviceId) {}

  int id() const { return deviceId; }

  M_driver_driver &getDriver() { return driver; }

  M_driver_status_t property(llvm::StringRef name, void *value);
  M_driver_status_t attribute(int32_t attribute, int32_t *value);
  M_driver_status_t canAccessPeer(const M_driver_device &peer,
                                  bool *out_can_access);
};

#endif // TARGET_LIB_HIP_DEVICE_H
