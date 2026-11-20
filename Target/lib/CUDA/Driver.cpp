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
#include "Driver.h"

#include <memory>

using M::Target::CUDA::cudaErrorStatus;
using M::Target::CUDA::nvmlErrorStatus;

M_driver_driver::~M_driver_driver() {
  rawCuda.unload();
  rawNvml.unload();
}

M_driver_status_t M_driver_driver::destroy(M_driver_driver_t driver) {
  delete driver;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_driver_t> M_driver_driver::create() {
  auto nvml = NVML::load();
  if (!nvml)
    return tl::unexpected(nvml.error());

  auto cuda = CUDALoader::load();
  if (!cuda) {
    nvml->unload();
    return tl::unexpected(cuda.error());
  }

  auto driver =
      std::make_unique<M_driver_driver>(std::move(*cuda), std::move(*nvml));

  if (nvmlReturn_t r = driver->rawNvml.nvmlInit_v2(); r != NVML_SUCCESS)
    driver->initStatus = nvmlErrorStatus(driver.get(), r, "nvmlInit_v2");
  else if (CUresult r = driver->rawCuda.cuInit(0); r != CUDA_SUCCESS)
    driver->initStatus = cudaErrorStatus(driver.get(), r, "cuInit");

  return driver.release();
}

DriverResult<int64_t> M_driver_driver::deviceCount() {
  unsigned int nvmlCount = 0;
  if (nvmlReturn_t r = rawNvml.nvmlDeviceGetCount_v2(&nvmlCount);
      r != NVML_SUCCESS)
    return tl::unexpected(nvmlErrorStatus(this, r, "nvmlDeviceGetCount_v2"));
  return static_cast<int64_t>(nvmlCount);
}
