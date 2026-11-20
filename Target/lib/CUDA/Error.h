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
#ifndef TARGET_LIB_CUDA_ERROR_H
#define TARGET_LIB_CUDA_ERROR_H

#include "HAL/M_driver_common.h"

#include "llvm/ADT/Twine.h"

#include <cuda.h>
#include <nvml.h>
#include <tl/expected.hpp>

#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 0

struct M_driver_driver;

namespace M::Target::CUDA {

M_driver_status_t cudaErrorStatus(M_driver_driver *driver, CUresult result,
                                  llvm::Twine const &context);

M_driver_status_t nvmlErrorStatus(M_driver_driver *driver, nvmlReturn_t result,
                                  llvm::Twine const &context);

} // namespace M::Target::CUDA

#define RETURN_IF_ERROR(stmt, error)                                           \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      return error;                                                            \
  } while (0)

#define QUICK_RETURN_IF_ERROR(stmt)                                            \
  do {                                                                         \
    M_driver_status_t status = (stmt);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      return status;                                                           \
  } while (0)

template <typename T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif // TARGET_LIB_CUDA_ERROR_H
