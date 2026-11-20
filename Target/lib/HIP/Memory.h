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

#ifndef TARGET_LIB_HIP_MEMORY_H
#define TARGET_LIB_HIP_MEMORY_H

#include "HAL/M_driver_memory.h"

#include "llvm/ADT/StringRef.h"

struct M_driver_memory {
  void *ptr;
  // False for wrapped (externally-owned) memory: freeing the handle must
  // not free the underlying region.
  bool owning = true;
  // True only for pinned host allocations, whose `ptr` is host-dereferenceable.
  bool hostAccessible = false;

  M_driver_status_t getProperty(llvm::StringRef name, void *value) {
    if (name == M_DRIVER_MEMORY_PROPERTY_ADDRESS) {
      *static_cast<void **>(value) = ptr;
    } else if (name == M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS) {
      if (!hostAccessible)
        return M_DRIVER_STATUS_INVALID_ARG;
      *static_cast<void **>(value) = ptr;
    } else {
      return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
    }
    return M_DRIVER_STATUS_SUCCESS;
  }
};

#endif // TARGET_LIB_HIP_MEMORY_H
