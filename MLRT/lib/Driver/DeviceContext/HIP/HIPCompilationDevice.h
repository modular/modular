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
// This file declares the HIP compilation device for compile-only mode.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPCOMPILATIONDEVICE_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPCOMPILATIONDEVICE_H

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <memory>
#include <string>

namespace M::Driver {

class DriverVersion;

/// HIP-specific compilation device for compile-only mode.
///
/// Provides HIP device information for compilation without requiring GPU
/// hardware or hipInit(). Unlike CUDA, HIP does not require a runtime
/// compilation step - KGEN handles all compilation through the LLVM AMDGPU
/// backend. This device primarily exists to store target architecture
/// information for the compiler.
///
/// Example usage:
///   auto device = HIPCompilationDevice::create("gfx942");
///   auto arch = device->getArchName();  // Returns "gfx942"
class HIPCompilationDevice : public CompilationDevice {
public:
  /// Create a HIP compilation device for the specified target architecture.
  ///
  /// \param targetArch Target architecture (e.g., "gfx942", "gfx90a")
  /// \return Compilation device or error
  static ErrorOr<std::unique_ptr<HIPCompilationDevice>>
  create(StringRef targetArch);

  ~HIPCompilationDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOr<const char *> getArchName() const override;

  /// HIP does not require runtime compilation like CUDA.
  ///
  /// For HIP, all compilation from LLVM IR to device code (.so with HSACO)
  /// is handled by KGEN using the LLVM AMDGPU backend and ld.lld linker.
  /// This method should not be called for HIP.
  ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel, int32_t optimizationLevel,
                  StringRef targetAccelerator) const override;

private:
  HIPCompilationDevice(std::string targetArch);
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_HIP_HIPCOMPILATIONDEVICE_H
