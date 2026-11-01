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
// This file declares the CompilationDevice interface for compile-only mode.
//
// CompilationDevice provides device functionality needed for compilation
// without requiring actual GPU hardware or driver initialization (cuInit()).
// This enables compiling GPU code on machines without GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_COMPILATIONDEVICE_H
#define MLRT_DRIVER_DEVICECONTEXT_COMPILATIONDEVICE_H

#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <cstdint>
#include <string>

namespace M::Driver {

/// A parsed driver version string (`major.minor.patch`). Lives with the
/// compilation-device layer so it is available regardless of which
/// DeviceContext implementation is active.
class DriverVersion {
public:
  explicit DriverVersion(std::string version);
  int major() const;
  int minor() const;
  int patch() const;
  std::string toString() const;

private:
  std::string version_;
  int major_;
  int minor_;
  int patch_;
};

/// CompilationDevice provides compilation functionality without GPU hardware.
///
/// This abstract base class defines the interface for device compilation
/// operations that do not require physical GPU access or driver initialization.
/// Implementations (CUDACompilationDevice, HIPCompilationDevice) provide
/// compilation for specific GPU architectures.
///
/// Key properties:
/// - Does NOT require cuInit() or GPU hardware
/// - Provides target architecture information for compilation
/// - Implements standalone compilation (e.g., NVPTX compiler for CUDA)
/// - Can be used by both real devices and virtual devices
///
/// Usage:
/// - CUDADevice uses CUDACompilationDevice for compilation operations
/// - VirtualDevice uses CUDACompilationDevice to compile without real GPU
class CompilationDevice {
public:
  using API = StringLiteral;

  virtual ~CompilationDevice() = default;

  /// Get the API type (cuda, hip, etc.)
  API api() const { return api_; }

  /// Get the target architecture name (e.g., "sm_80", "gfx90a")
  const std::string &targetArch() const { return targetArch_; }

  /// Get the API version supported by this compilation device
  virtual ErrorOr<int> getApiVersion() const = 0;

  /// Get the driver version (may be simulated for compile-only devices)
  virtual ErrorOr<DriverVersion &> getDriverVersion() const = 0;

  /// Get the architecture name string (e.g., "sm_80")
  virtual ErrorOr<const char *> getArchName() const = 0;

  /// Compile intermediate representation (IR) to device binary.
  ///
  /// For CUDA: Compiles PTX to CUBIN using libnvptxcompiler
  /// For HIP: Compiles to device code using HIP compiler
  ///
  /// This method does NOT require GPU hardware or driver initialization.
  ///
  /// \param ir The intermediate representation code (e.g., PTX for CUDA)
  /// \param debugLevel Debug level for compilation
  /// \param optimizationLevel Optimization level (-1 for default)
  /// \param targetAccelerator Target accelerator override (e.g., "sm_80")
  /// \return Compiled binary code or error
  virtual ErrorOr<M::BufferRef>
  compileFunction(StringRef ir, StringRef debugLevel, int32_t optimizationLevel,
                  StringRef targetAccelerator = "") const = 0;

  /// \param api The api for which to create a compilation device. Accepts
  ///        either a device-API name ("cuda", "hip") or the Mojo
  ///        `--target-accelerator` vendor name for the same device ("nvidia",
  ///        "amdgpu"/"amd"), so a caller holding a target-accelerator value
  ///        can pass it through unchanged. `api()` on the result is always
  ///        the device-API name.
  /// \return Compilation device or error
  static ErrorOr<std::unique_ptr<CompilationDevice>> create(StringRef api,
                                                            StringRef arch);

protected:
  CompilationDevice(API api, std::string targetArch)
      : api_(api), targetArch_(std::move(targetArch)) {}

private:
  API api_;
  std::string targetArch_;
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_COMPILATIONDEVICE_H
