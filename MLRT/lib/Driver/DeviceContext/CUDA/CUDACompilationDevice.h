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
// This file declares the CUDA compilation device for compile-only mode.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDEVICE_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDEVICE_H

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <memory>
#include <string>

namespace M::Driver {

class DriverVersion;

/// CUDA-specific compilation device for compile-only mode.
///
/// Provides CUDA compilation functionality without requiring GPU hardware
/// or cuInit(). Uses libnvptxcompiler to compile PTX to CUBIN.
///
/// Example usage:
///   auto device = CUDACompilationDevice::create("sm_80");
///   auto cubin = device->compileFunction(ptx, "0", 3, "sm_80");
class CUDACompilationDevice : public CompilationDevice {
public:
  /// Create a CUDA compilation device for the specified target architecture.
  ///
  /// \param targetArch Target architecture (e.g., "sm_80", "sm_90", "sm_100")
  /// \return Compilation device or error
  static ErrorOr<std::unique_ptr<CUDACompilationDevice>>
  create(StringRef targetArch);

  ~CUDACompilationDevice() override;

  ErrorOr<int> getApiVersion() const override;
  ErrorOr<DriverVersion &> getDriverVersion() const override;
  ErrorOr<const char *> getArchName() const override;

  /// Compile PTX to CUBIN using libnvptxcompiler.
  ///
  /// This method does NOT require GPU hardware or cuInit(). It uses the
  /// standalone NVPTX compiler library.
  ErrorOr<M::BufferRef>
  compileFunction(StringRef ptx, StringRef debugLevel,
                  int32_t optimizationLevel,
                  StringRef targetAccelerator = "") const override;

  /// Setup compilation options for NVPTX compiler.
  ///
  /// This is a utility function that can be used by both CUDACompilationDevice
  /// and CUDADeviceContext.
  static SmallVector<std::string> setupCompilationOptionsNVPTXCompiler(
      StringRef debugLevel, int32_t optimizationLevel, StringRef gpuName);

protected:
  CUDACompilationDevice(std::string targetArch);
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDEVICE_H
