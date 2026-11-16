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
// The architecture of a device as numbers, so that a caller compares
// architectures instead of parsing their names.
//
// The version comes only from the device's API and architecture name, never
// from the hardware, so a virtual (compile-only) device configured as
// `sm_100a` or `gfx950` reports the same version as a real device of that
// architecture. A target alias such as `mi355x` is not an architecture name
// and has no version.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_CAPI_ARCHITECTUREVERSION_H
#define MLRT_DRIVER_CAPI_ARCHITECTUREVERSION_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace M::Driver {

class Device;

/// The architecture version of a device. The fields follow the vendor's own
/// numbering, so compare them only between devices with the same API:
///
/// | API     | Architecture name   | major | minor | stepping |
/// |---------|---------------------|-------|-------|----------|
/// | `cuda`  | `sm_90a`, `sm_100a` | 9, 10 | 0, 0  | 0, 0     |
/// | `hip`   | `gfx942`, `gfx90a`  | 9, 9  | 4, 0  | 2, 10    |
/// | `metal` | `4`, `5-metal4`     | 4, 5  | 0, 0  | 0, 0     |
///
/// CUDA reports the compute capability, and drops the `a` and `f` feature-set
/// suffixes. HIP reports the AMDGPU processor version
/// `gfx<major><minor><stepping>`, whose stepping is a hexadecimal digit, and
/// drops target features such as `:xnack-`. Metal reports the Apple silicon
/// generation (M5 is 5), and drops the Metal language suffix.
///
/// The type has no ordering because a higher version is not always a newer
/// or more capable part: `gfx1100` (RDNA3) sorts above `gfx950` (CDNA4), and
/// `sm_120` lacks features of `sm_100`.
struct ArchitectureVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;

  bool operator==(const ArchitectureVersion &) const = default;
};

/// Parses `architectureName` as the architecture name a device of `api`
/// reports. Fails for the CPU, for an API with no known numbering, and for a
/// name not in the form the table above shows.
ErrorOr<ArchitectureVersion>
parseArchitectureVersion(llvm::StringRef api, llvm::StringRef architectureName);

/// Returns the architecture version of `device`.
ErrorOr<ArchitectureVersion> getArchitectureVersion(const Device &device);

} // namespace M::Driver

#endif // MLRT_DRIVER_CAPI_ARCHITECTUREVERSION_H
