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

#include "Error.h"
#include "Driver.h"

#include "llvm/ADT/Twine.h"

#include <cstdint>

namespace M::Target::HIP {

M_driver_status_t hipErrorStatus(M_driver_driver *driver, hipError_t result,
                                 llvm::Twine const &context) {
  HIPApi &hip = driver->hip();
  const char *name = hip.hipGetErrorName(result);
  const char *desc = hip.hipGetErrorString(result);

  if (name != nullptr && desc != nullptr)
    return driver->status.create(context + ": " + name + " (" + desc + ")");
  if (name != nullptr)
    return driver->status.create(context + ": " + name);
  if (desc != nullptr)
    return driver->status.create(context + ": " + desc);
  return driver->status.create(context + ": hipError_t " +
                               llvm::Twine(static_cast<int64_t>(result)));
}

} // namespace M::Target::HIP
