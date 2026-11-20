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
// This file implements the Compilation Device.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "CUDA/CUDACompilationDevice.h"
#include "HIP/HIPCompilationDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"
#include "Target/DriverPlugin.h"

namespace M::Driver {

//
// ===== DriverVersion ===== //
//

DriverVersion::DriverVersion(std::string version) : version_(version) {
  std::sscanf(version.c_str(), "%d.%d.%d", &major_, &minor_, &patch_);
}

int DriverVersion::major() const { return major_; }

int DriverVersion::minor() const { return minor_; }

int DriverVersion::patch() const { return patch_; }

MODULAR_CXX_EXPORT std::string DriverVersion::toString() const {
  return version_;
}

MODULAR_CXX_EXPORT ErrorOr<std::unique_ptr<CompilationDevice>>
CompilationDevice::create(StringRef requestedAPI, StringRef arch) {
  const StringRef api = canonicalDeviceAPI(requestedAPI);

  // Branch based on the configured API (not the requested API which is often
  // generic "gpu")
  if (api == "cuda") {
    auto compilationDeviceOrErr = CUDACompilationDevice::create(arch);
    if (compilationDeviceOrErr.isError()) {
      return Error(
          llvm::Twine(
              "Failed to create CUDA compilation device for virtual mode: ") +
          compilationDeviceOrErr.getError());
    }

    return std::move(*compilationDeviceOrErr);
  } else if (api == "hip") {
    // Create HIP compilation device
    // Unlike CUDA, HIP does not require runtime compilation (PTX→CUBIN).
    // All compilation is handled by KGEN through the LLVM AMDGPU backend.
    // However, we still need a CompilationDevice to store the target
    // architecture information.
    auto compilationDeviceOrErr = HIPCompilationDevice::create(arch);
    if (compilationDeviceOrErr.isError()) {
      return Error(
          llvm::Twine(
              "Failed to create HIP compilation device for virtual mode: ") +
          compilationDeviceOrErr.getError());
    }

    return std::move(*compilationDeviceOrErr);
  } else if (api == "metal") {
    // TODO: Implement MetalCompilationDevice for Apple GPU compilation
    return Error("Metal virtual device mode not yet implemented. "
                 "Only CUDA is currently supported in compile-only mode.");
  }

  // Generic plugin-based compilation device (no hardcoded API names).
  {
    auto compilationDeviceOr = Plugin::getCompilationDevice(api, arch);
    if (!compilationDeviceOr.isError())
      return std::move(*compilationDeviceOr);
    // Fall through to unknown API error if no plugin handles this.
  }

  return Error(llvm::Twine("Unknown virtual device API: ") + requestedAPI);
}

} // namespace M::Driver
