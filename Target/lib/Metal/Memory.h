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

#ifndef TARGET_LIB_METAL_MEMORY_H
#define TARGET_LIB_METAL_MEMORY_H

#include "HAL/M_driver_common.h"
#include "MetalCpp.h"

#include "llvm/ADT/StringRef.h"

struct M_driver_memory {
  MTL::Buffer *mBuffer;
  // Owning handles release the buffer on free; `unwrap` clears this so a
  // borrowed handle's free releases only the wrapper.
  bool owning = true;
  // Sub-range offset within `mBuffer`: every derived address must add it, or a
  // wrapped sub-range would alias the buffer base. Zero for whole-buffer
  // allocs.
  uint64_t offset = 0;

  M_driver_status_t getProperty(llvm::StringRef name, void *value);
};

#endif // TARGET_LIB_METAL_MEMORY_H
