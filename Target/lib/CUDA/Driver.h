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

#ifndef TARGET_LIB_CUDA_DRIVER_H
#define TARGET_LIB_CUDA_DRIVER_H

#include "Error.h"
#include "Loader/CUDALoader.h"
#include "Loader/NVML.h"
#include "Target/lib/Support/Utils.h"

using M::Target::CustomStatus;
using M::Target::CUDA::CUDALoader;
using M::Target::CUDA::NVML;

struct M_driver_driver {
  CUDALoader rawCuda;
  NVML rawNvml;
  CustomStatus status;

  M_driver_status_t initStatus = M_DRIVER_STATUS_SUCCESS;

  [[nodiscard]] static DriverResult<M_driver_driver_t> create();
  static M_driver_status_t destroy(M_driver_driver_t driver);

  M_driver_driver(CUDALoader &&cuda, NVML &&nvml)
      : rawCuda(std::move(cuda)), rawNvml(std::move(nvml)),
        status(M_DRIVER_NUM_CUSTOM_STATUS_CODES) {}
  ~M_driver_driver();
  M_driver_driver(const M_driver_driver &) = delete;
  M_driver_driver &operator=(const M_driver_driver &) = delete;

  DriverResult<int64_t> deviceCount();
};

#endif // TARGET_LIB_CUDA_DRIVER_H
