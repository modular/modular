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

#include "Driver.h"

#include <memory>

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_driver_t> M_driver_driver::create() {
  DriverResult<HIPApi> api = HIPApi::load();
  if (!api)
    return tl::unexpected(api.error());

  auto driver = std::make_unique<M_driver_driver>(std::move(*api));

  if (hipError_t r = driver->hip().hipInit(0); r != hipSuccess)
    driver->initStatus = hipErrorStatus(driver.get(), r, "hipInit");

  return driver.release();
}

DriverResult<int64_t> M_driver_driver::deviceCount() {
  int countAsInt = 0;
  if (hipError_t r = rawHipApi.hipGetDeviceCount(&countAsInt); r != hipSuccess)
    return tl::unexpected(hipErrorStatus(this, r, "hipGetDeviceCount"));
  return static_cast<int64_t>(countAsInt);
}
