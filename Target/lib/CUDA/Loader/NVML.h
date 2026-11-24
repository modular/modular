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
// This file declares the NVML API interface.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_NVML_H
#define TARGET_LIB_CUDA_NVML_H

#include "Target/lib/CUDA/Error.h"

#include <nvml.h>

namespace M::Target::CUDA {

struct NVML {
  void *lib = nullptr;

#define NVML_FN(name) decltype(&::name) name = nullptr;
#include "NVML.inc"
#undef NVML_FN

  NVML(const NVML &) = delete;
  NVML &operator=(const NVML &) = delete;
  NVML(NVML &&) = default;
  NVML &operator=(NVML &&) = default;
  NVML() = default;

  static DriverResult<NVML> load();
  void unload();

  /// Returns the NVML driver version string, or nullptr on failure.
  const char *getDriverVersion() const;
};

} // namespace M::Target::CUDA

#endif // TARGET_LIB_CUDA_NVML_H
