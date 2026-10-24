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
// This file declares the CUDA compilation driver interface.
//
// CUDACompilationDriver provides access to NVPTX compiler functions
// (libNVPTX.so) without requiring GPU hardware or cuInit().
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDRIVER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDRIVER_H

#include "Support/ErrorOr.h"
#include "Support/Preprocessor.h"
#include "Support/max_license/Tracy/Tracy.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <stddef.h>

// The opaque compiler handle. Declared here rather than pulled in from
// nvPTXCompiler.h: the library is resolved at runtime, so nothing links
// against it and its headers are not on the include path.
using nvPTXCompilerHandle = struct nvPTXCompiler *;

namespace M::Driver {

/// Check if NVPTX API logging is enabled at runtime via MLRT_CUDA_DEBUG env
/// var.
inline bool isNVPTXDebugLoggingEnabled() {
  static const bool enabled =
      llvm::sys::Process::GetEnv("MLRT_CUDA_DEBUG").has_value();
  return enabled;
}

/// Runtime logging macro for NVPTX API calls.
#define CUDA_NVPTX_CALL_LOG()                                                  \
  if (isNVPTXDebugLoggingEnabled()) {                                          \
    llvm::errs() << "[NVPTX] " << __FUNCTION__ << "\n";                        \
  }

/// CUDACompilationDriver provides access to NVPTX compiler functions.
///
/// This class dlopens the NVPTX compiler library and provides methods to
/// compile PTX to CUBIN without requiring GPU hardware or cuInit().
///
/// The NVPTX compiler library is standalone and does not depend on
/// libcuda.so or any GPU driver initialization.
class CUDACompilationDriver {
public:
  enum Result : unsigned int {
    SUCCESS = 0,
    ERROR = 1,
  };

// Define wrapper member function with runtime logging.
#define NVPTX_FUNC(name, ret, ...)                                             \
  ret name(EXPAND_PARAMS(__VA_ARGS__)) const {                                 \
    CUDA_NVPTX_CALL_LOG();                                                     \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_GREEN);                            \
    ret result = name##_(EXPAND_ARGS(__VA_ARGS__));                            \
    if (isNVPTXDebugLoggingEnabled()) {                                        \
      llvm::errs() << "[NVPTX] " << #name << " -> " << result << "\n";         \
    }                                                                          \
    return result;                                                             \
  }
#include "CUDACompilationDriver.inc"
#undef NVPTX_FUNC

  static ErrorOr<const CUDACompilationDriver &> get();

private:
  static ErrorOr<std::unique_ptr<CUDACompilationDriver>> allocAndInitDriver();

#define NVPTX_FUNC(name, ret, ...)                                             \
  using t##name = ret(__VA_ARGS__);                                            \
  t##name *name##_;
#include "CUDACompilationDriver.inc"
#undef NVPTX_FUNC
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_CUDACOMPILATIONDRIVER_H
