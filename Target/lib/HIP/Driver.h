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

#ifndef TARGET_LIB_HIP_DRIVER_H
#define TARGET_LIB_HIP_DRIVER_H

#include "Error.h"
#include "HIPApi.h"
#include "Target/lib/Support/Utils.h"

using M::Target::CustomStatus;

struct M_driver_driver {
private:
  HIPApi rawHipApi;

public:
  CustomStatus status;

  M_driver_status_t initStatus = M_DRIVER_STATUS_SUCCESS;

  [[nodiscard]] static DriverResult<M_driver_driver_t> create();

  static M_driver_status_t destroy(M_driver_driver_t driver) {
    delete driver;
    return M_DRIVER_STATUS_SUCCESS;
  }

  M_driver_driver(HIPApi &&rawHipApi)
      : rawHipApi(std::move(rawHipApi)),
        status(M_DRIVER_NUM_CUSTOM_STATUS_CODES) {}

  ~M_driver_driver() { rawHipApi.unload(); }

  HIPApi &hip() { return rawHipApi; }

  DriverResult<int64_t> deviceCount();
};

#endif // TARGET_LIB_HIP_DRIVER_H
