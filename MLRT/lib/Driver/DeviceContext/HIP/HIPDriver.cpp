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
// This file implements the HIP driver.
//
//===----------------------------------------------------------------------===//

#include "HIPDriver.h"

#include "Support/DLFcn.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "Support/Profiling/Ranges.h"
#include "llvm/ADT/Twine.h"

#include <stdlib.h>

namespace M::Driver {

#if MODULAR_LINUX
const char HIPDriver::lib_name[] = "libamdhip64.so";
const char hsa_lib_name[] = "libhsa-runtime64.so";
#else
const char HIPDriver::lib_name[] = "";
const char hsa_lib_name[] = "";
#endif

#define GET_HSA_FUNC(name)                                                     \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(hsa_lib, #name));        \
  // Note: We don't fail if HSA functions are not available

ErrorOr<std::unique_ptr<HIPDriver>> HIPDriver::allocAndInitDriver() {
  if (strcmp("", lib_name) == 0) {
    return Error(llvm::Twine("Failed to open HIP library: "
                             "hip is not supported on this operating system."));
  }

  // Activate on-demand profiling (libkineto/rocprofiler-sdk) BEFORE the HIP
  // runtime loads: rocprofiler-sdk only accepts in-process tool registration
  // until the HIP/HSA runtime initializes, so this is the AMD mirror of the
  // post-context-creation call in CUDADeviceContext.cpp — see
  // activatePendingTrace() in Support/Profiling/Ranges.h for the contract.
  // Cheap no-op unless a profiler was requested for this process (enable(), the
  // max-debug.profiling-enabled auto-start knob, MODULAR_PROFILER_PLUGIN, or
  // Dynolog daemon mode).
  M::Profiling::activatePendingTrace();

  // Attempt to open the shared library of the HIP driver.
  // RTLD_GLOBAL allows UCX ROCm backend to also use these loaded symbols.
  void *dl_lib = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
  if (dl_lib == nullptr) {
    return Error(llvm::Twine("Failed to open library \"") + lib_name +
                 "\": " + dlerror());
  }

  // Allocate a driver and register all of the used shared library entry points.
  auto driver = std::make_unique<HIPDriver>();

#define HIP_FUNC(name, ...)                                                    \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dl_lib, #name));         \
  if (driver->name##_ == nullptr) {                                            \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + lib_name + "\"");                                \
  }

// Optional entries: store nullptr on dlsym miss without returning an error.
#define HIP_OPT_FUNC(name, ...)                                                \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dl_lib, #name));

#include "HIPDriver.inc"
#undef HIP_OPT_FUNC
#undef HIP_FUNC

  driver->hipGetLastError_ =
      reinterpret_cast<thipGetLastError *>(dlsym(dl_lib, "hipGetLastError"));
  if (driver->hipGetLastError_ == nullptr) {
    return Error(
        llvm::Twine("Failed to find symbol \"hipGetLastError\" in \"") +
        lib_name + "\"");
  }

  int driverVersion = 0;
  if (driver->hipDriverGetVersion(&driverVersion) != SUCCESS) {
    return Error("hipDriverGetVersion failed");
  }

  // Old versions of hipMemcpyBatchAsync failed when passed a valid attribute.
  // These versions also implement the API as a loop over individual memcpy
  // calls, so treat these platforms as not supporting the API.
  if (driverVersion < 71400000) {
    driver->hipMemcpyBatchAsync_ = nullptr;
  }

  // Try to load HSA Runtime symbols (optional - for getting device
  // architecture).
  // RTLD_GLOBAL allows UCX ROCm backend to also use these loaded symbols.
  void *hsa_lib = dlopen(hsa_lib_name, RTLD_NOW | RTLD_GLOBAL);
  if (hsa_lib != nullptr) {
    GET_HSA_FUNC(hsa_init);
    GET_HSA_FUNC(hsa_shut_down);
    GET_HSA_FUNC(hsa_iterate_agents);
    GET_HSA_FUNC(hsa_agent_get_info);
    GET_HSA_FUNC(hsa_agent_iterate_isas);
    GET_HSA_FUNC(hsa_isa_get_info_alt);
  } else {
    // HSA runtime not available - set all function pointers to nullptr
    driver->hsa_init_ = nullptr;
    driver->hsa_shut_down_ = nullptr;
    driver->hsa_iterate_agents_ = nullptr;
    driver->hsa_agent_get_info_ = nullptr;
    driver->hsa_agent_iterate_isas_ = nullptr;
    driver->hsa_isa_get_info_alt_ = nullptr;
  }

  // Initialize the driver and return the newly allocated driver if successful.
  Result err = driver->hipInit(0);
  if (err != SUCCESS) {
    const char *errName = nullptr;
    const char *errStr = nullptr;
    driver->hipDrvGetErrorName(err, &errName);
    driver->hipDrvGetErrorString(err, &errStr);

    return Error(llvm::Twine("Call to hipInit failed with error code: ") +
                 llvm::Twine(err) + ", error name: " + llvm::Twine(errName) +
                 ", error string: " + llvm::Twine(errStr));
  }

  return driver;
}

ErrorOr<const HIPDriver &> HIPDriver::get() {
  static ErrorOr<std::unique_ptr<HIPDriver>> errOrDriver = allocAndInitDriver();

  if (errOrDriver) {
    return Error(errOrDriver.getError());
  }
  return *errOrDriver.get();
}

} // namespace M::Driver
