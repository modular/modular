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
// Per-vendor queries for a local accelerator's arch. They are in
// libMDriverCompilation because the compiler and the device contexts both use
// them.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_ACCELERATORARCH_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_ACCELERATORARCH_H

#include "HIP/HIPDriver.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/PlatformUtils.h"
#include "Support/SymbolExport.h"

#include <optional>
#include <string>

#if MODULAR_MACOS
namespace MTL {
class Device;
} // namespace MTL
#endif

namespace M::Driver {

/// The `vendor:arch` of device 0 of the first accelerator plugin in
/// `MODULAR_DRIVER_PLUGINS` that implements the HAL C ABI, or "" when that
/// plugin has no device. `std::nullopt` when the variable names no such
/// plugin, so the caller falls back to the vendor libraries.
std::optional<std::string> getHALPluginAcceleratorArch();

/// For example `sm_80`, or `sm_90a` from sm_90 on.
MODULAR_CXX_EXPORT std::string cudaCCToSM(int major, int minor);

/// Reads device 0 through NVML, so it needs no CUDA context.
ErrorOr<std::string> getFirstCUDAArchName();

MODULAR_CXX_EXPORT ErrorOr<std::string> getHIPArchName(hipDevice_t device);

MODULAR_CXX_EXPORT Error formatHipError(const char *file, int line,
                                        const char *func,
                                        HIPDriver::Result result,
                                        std::string extra_msg = "");

#define HIP_CHECK_ERR(hip_call)                                                \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      return formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);      \
    }                                                                          \
  } while (0)

#if MODULAR_MACOS
/// The Apple GPU generation in a Metal device name, for example 4 for an M4.
MODULAR_CXX_EXPORT ErrorOr<int> getMetalComputeCapability(StringRef name);

/// Queries the system default device when `device` is null.
MODULAR_CXX_EXPORT ErrorOr<std::string> getMetalArchName(MTL::Device *device);
#endif

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_ACCELERATORARCH_H
