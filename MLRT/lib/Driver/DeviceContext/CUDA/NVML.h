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
// This file declares the NVML API interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_NVML_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_NVML_H

#include "Support/ErrorOr.h"
#include "Support/Preprocessor.h"
#include "Support/max_license/Tracy/Tracy.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <thread>

#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE 16

namespace M::Driver {

using nvmlDevice_t = struct nvmlDevice_st *;

/// Check if NVML API logging is enabled at runtime via MLRT_CUDA_DEBUG env var.
inline bool isNVMLDebugLoggingEnabled() {
  static const bool enabled =
      llvm::sys::Process::GetEnv("MLRT_CUDA_DEBUG").has_value();
  return enabled;
}

/// Runtime logging macro for NVML API calls.
#define CUDA_NVML_CALL_LOG()                                                   \
  if (isNVMLDebugLoggingEnabled()) {                                           \
    llvm::errs() << "[NVML] " << __FUNCTION__ << "\n";                         \
  }

struct nvmlPciInfo_t {
  char busIdLegacy[NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
};

struct nvmlMemory_t {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct nvmlProcessInfo_t {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
};

// Throttle reason bitmask constants.
static constexpr unsigned long long NVML_CLK_THROTTLE_REASON_HW_SLOWDOWN =
    0x8ULL;
static constexpr unsigned long long
    NVML_CLK_THROTTLE_REASON_HW_THERMAL_SLOWDOWN = 0x40ULL;
static constexpr unsigned long long
    NVML_CLK_THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN = 0x80ULL;

// ECC error type and counter type constants.
static constexpr int NVML_MEMORY_ERROR_TYPE_UNCORRECTABLE = 1;
static constexpr int NVML_VOLATILE_ECC = 0;

// MIG mode, as reported by nvmlDeviceGetMigMode.
static constexpr unsigned int NVML_DEVICE_MIG_DISABLE = 0;
static constexpr unsigned int NVML_DEVICE_MIG_ENABLE = 1;

class NVML {
public:
  enum Result : unsigned int {
    SUCCESS = 0,
    NOT_SUPPORTED = 3,
    NO_PERMISSION = 4,
    INSUFFICIENT_SIZE = 7,
  };

  ~NVML();

  ErrorOr<std::string> GetDriverVersion() const;

// Define wrapper member function with runtime logging.
#define NVML_FUNC(name, ret, ...)                                              \
  ret name(EXPAND_PARAMS(__VA_ARGS__)) const {                                 \
    CUDA_NVML_CALL_LOG();                                                      \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_GREEN);                            \
    ret result = name##_(EXPAND_ARGS(__VA_ARGS__));                            \
    if (isNVMLDebugLoggingEnabled()) {                                         \
      llvm::errs() << "[NVML] " << #name << " -> " << result << "\n";          \
    }                                                                          \
    return result;                                                             \
  }

#define NVML_FUNC_NO_PARAMS(name, ret)                                         \
  ret name() const {                                                           \
    CUDA_NVML_CALL_LOG();                                                      \
    TRACY_ZONE_SCOPED_NC(#name, TRACY_COLOR_GREEN);                            \
    ret result = name##_();                                                    \
    if (isNVMLDebugLoggingEnabled()) {                                         \
      llvm::errs() << "[NVML] " << #name << " -> " << result << "\n";          \
    }                                                                          \
    return result;                                                             \
  }
#include "NVML.inc"
#undef NVML_FUNC_NO_PARAMS
#undef NVML_FUNC

  static ErrorOr<const NVML &> get();

private:
  static ErrorOr<std::unique_ptr<NVML>> allocAndInitNVML();
  static const char lib_name[];

#define NVML_FUNC(name, ret, ...)                                              \
  using t##name = ret(__VA_ARGS__);                                            \
  t##name *name##_;

#define NVML_FUNC_NO_PARAMS(name, ret)                                         \
  using t##name = ret();                                                       \
  t##name *name##_;
#include "NVML.inc"
#undef NVML_FUNC_NO_PARAMS
#undef NVML_FUNC
};

} // namespace M::Driver
#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_CUDA_NVML_H
