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

#ifndef TARGET_LIB_CUDA_DEVICE_H
#define TARGET_LIB_CUDA_DEVICE_H

#include "Driver.h"
#include "Error.h"

#include <cuda.h>

struct M_driver_device {
  M_driver_driver &driver;
  CUdevice deviceId;

  [[nodiscard]] static DriverResult<M_driver_device_t>
  create(M_driver_driver &driver, int64_t deviceId);

  static M_driver_status_t destroy(M_driver_device_t device);

  explicit M_driver_device(M_driver_driver &driver, CUdevice id)
      : driver(driver), deviceId(id) {}

  M_driver_status_t property(llvm::StringRef name, void *value);
  M_driver_status_t attribute(int32_t attribute, int32_t *value);
  // False without native atomics, which spin-barrier collectives need.
  M_driver_status_t canAccessPeer(const M_driver_device &peer,
                                  bool *out_can_access) const;
};

#endif // TARGET_LIB_CUDA_DEVICE_H
