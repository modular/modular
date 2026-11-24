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

#ifndef TARGET_LIB_CPU_DRIVER_H
#define TARGET_LIB_CPU_DRIVER_H

#include "Common.h"
#include "Device.h"
#include "Target/lib/Support/Utils.h"

using M::Target::CustomStatus;

struct M_driver_driver {
  M_driver_driver() : device(this) {}

  [[nodiscard]] static M_driver_driver_t create();
  static void destroy(M_driver_driver_t driver);

  // The CPU plugin exposes exactly one CPU device.
  M_driver_device device;
  CustomStatus status{M_DRIVER_NUM_CUSTOM_STATUS_CODES};
};

#endif // TARGET_LIB_CPU_DRIVER_H
