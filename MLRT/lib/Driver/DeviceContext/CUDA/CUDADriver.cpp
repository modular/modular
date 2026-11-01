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
// This file implements the CUDA driver.
//
//===----------------------------------------------------------------------===//

#include "CUDADriver.h"

#include "Support/DLFcn.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <stdlib.h>
#include <string.h>

namespace M::Driver {

#if MODULAR_LINUX
const char CUDADriver::lib_name[] = "libcuda.so.1";
#else
const char CUDADriver::lib_name[] = "";
#endif

ErrorOr<std::unique_ptr<CUDADriver>> CUDADriver::allocAndInitDriver() {
  if (strcmp("", lib_name) == 0) {
    return Error(
        llvm::Twine("Failed to open CUDA library: "
                    "cuda is not supported on this operating system."));
  }

  // We use both NVML and CUDA Driver to access devices. NVML discovers devices
  // in the order of PCI bus ID, so we want to force CUDA to enumerate in the
  // same order.
  // See:
  // [1] Documentation for nvmlDeviceGetHandleByIndex_v2 (non-deterministic
  //     device-ordering between reboots)
  // [2]
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables
#ifdef _WIN32
  _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", /*overwrite=*/1);
#endif

  // Attempt to open the shared library of the CUDA driver.
  // We use RTLD_GLOBAL to allow UCX CUDA backend to also use these loaded CUDA
  // symbols.
  void *dlLib = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
  if (dlLib == nullptr) {
    return Error(llvm::Twine("Failed to open library \"") + lib_name + "\"");
  }

  // Allocate a driver and register all of the used shared library entry points.
  auto driver = std::make_unique<CUDADriver>();

#define CUDA_FUNC(name, ...)                                                   \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dlLib, #name));          \
  if (driver->name##_ == nullptr) {                                            \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + lib_name + "\"");                                \
  }

// Optional entries: store nullptr on dlsym miss without returning an error.
// This allows the driver to initialize on CUDA < 12.8 hosts that do not
// export cuMemcpyBatchAsync.
#define CUDA_OPT_FUNC(name, ...)                                               \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(dlLib, #name));

#include "CUDADriver.inc"
#undef CUDA_OPT_FUNC
#undef CUDA_FUNC

  // Initialize the driver and return the newly allocated driver if successful.
  Result err = driver->Init(0);
  if (err != SUCCESS) {
    return Error(llvm::Twine("Call to cuInit failed with ") + llvm::Twine(err));
  }

  return driver;
}

ErrorOr<const CUDADriver &> CUDADriver::get() {
  static ErrorOr<std::unique_ptr<CUDADriver>> errOrDriver =
      allocAndInitDriver();

  if (errOrDriver) {
    return Error(errOrDriver.getError());
  }

  return *errOrDriver.get();
}

} // namespace M::Driver
