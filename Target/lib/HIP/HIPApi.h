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
// Wraps the HIP C API as loaded through `dlopen`.
// Types come from HIP itself, but main HIP libraries only exist at runtime.
// Using dlopen eliminates link-time dependency on libamdhip64.so.
//
// To add a new HIP API call, add a single entry to HIP_FORALL_FUNCTIONS.

#ifndef HIPAPI_H
#define HIPAPI_H

#include "Error.h"
// Suppress C++ template overloads for hipMalloc, hipHostMalloc, etc.
// so that decltype(&::name) resolves unambiguously to the C signatures.
// This comes from HIP, so nothing we can do about the fact
// it conflicts with reserved string space for defs from C++
// NOLINTNEXTLINE
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "hip/hip_runtime_api.h"

#include <dlfcn.h>
#include <stdio.h>

// Add a `HIP_FN(<fn_name>)` for each fn to dynamically link and
// hold a member for. `HIP_FN` derives type information from decltype,
// so no additional work needs to be done to keep types in sync.
#define HIP_FORALL_FUNCTIONS(HIP_FN)                                           \
  HIP_FN(hipInit)                                                              \
  HIP_FN(hipGetDeviceCount)                                                    \
  HIP_FN(hipDeviceGet)                                                         \
  HIP_FN(hipDeviceReset)                                                       \
  HIP_FN(hipSetDevice)                                                         \
  HIP_FN(hipGetDevice)                                                         \
  HIP_FN(hipDeviceGetAttribute)                                                \
  HIP_FN(hipDeviceGetName)                                                     \
  HIP_FN(hipMemGetInfo)                                                        \
  HIP_FN(hipMalloc)                                                            \
  HIP_FN(hipFree)                                                              \
  HIP_FN(hipHostMalloc)                                                        \
  HIP_FN(hipHostFree)                                                          \
  HIP_FN(hipModuleLoadData)                                                    \
  HIP_FN(hipModuleGetFunction)                                                 \
  HIP_FN(hipModuleLaunchKernel)                                                \
  HIP_FN(hipModuleUnload)                                                      \
  HIP_FN(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)                   \
  HIP_FN(hipStreamCreateWithPriority)                                          \
  HIP_FN(hipStreamDestroy)                                                     \
  HIP_FN(hipStreamSynchronize)                                                 \
  HIP_FN(hipStreamWaitValue64)                                                 \
  HIP_FN(hipLaunchHostFunc)                                                    \
  HIP_FN(hipMemcpyAsync)                                                       \
  HIP_FN(hipMemcpyPeerAsync)                                                   \
  HIP_FN(hipMemcpy)                                                            \
  HIP_FN(hipMemcpyPeer)                                                        \
  HIP_FN(hipDeviceCanAccessPeer)                                               \
  HIP_FN(hipDeviceEnablePeerAccess)                                            \
  HIP_FN(hipDeviceGetDefaultMemPool)                                           \
  HIP_FN(hipMemPoolSetAccess)                                                  \
  HIP_FN(hipGetLastError)                                                      \
  HIP_FN(hipMemsetAsync)                                                       \
  HIP_FN(hipMemsetD16Async)                                                    \
  HIP_FN(hipMemsetD32Async)                                                    \
  HIP_FN(hipEventCreateWithFlags)                                              \
  HIP_FN(hipEventDestroy)                                                      \
  HIP_FN(hipEventRecord)                                                       \
  HIP_FN(hipStreamWaitEvent)                                                   \
  HIP_FN(hipEventSynchronize)                                                  \
  HIP_FN(hipEventQuery)                                                        \
  HIP_FN(hipGetErrorName)                                                      \
  HIP_FN(hipGetErrorString)

struct HIPApi {
  // dlopen handle to loaded libamdhip64.so.
  void *lib = nullptr;

  // Declare member fn ptrs, derive types from corresponding decls in HIP
  // headers
#define DECLARE_FN_PTR(name) decltype(&::name) name = nullptr;
  HIP_FORALL_FUNCTIONS(DECLARE_FN_PTR)
#undef DECLARE_FN_PTR

  // `hipGetDeviceProperties` is ABI-versioned on ROCm 6 (the real export is
  // `hipGetDevicePropertiesR0600`) so we don't use the HIP_FN macro so we can
  // fall back to the older version.
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *, int) = nullptr;

  // HIPApi owns the dlopen handle; copying would produce two owners that both
  // call dlclose on destruction.
  HIPApi(const HIPApi &) = delete;
  HIPApi &operator=(const HIPApi &) = delete;
  HIPApi(HIPApi &&) = default;
  HIPApi &operator=(HIPApi &&) = default;
  HIPApi() = default;

  /// Try to dlopen libamdhip64.so and resolve all symbols.
  /// Returns a HIPApi on success, or an error status on failure.
  static DriverResult<HIPApi> load() {
    void *lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
      fprintf(stderr, "HIPPlugin: dlopen(libamdhip64.so) failed: %s\n",
              dlerror());
      return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);
    }

    auto api = HIPApi();
    api.lib = lib;

#define LOAD_SYM(name)                                                         \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(lib, #name));          \
  if (!api.name) {                                                             \
    fprintf(stderr, "HIPPlugin: dlsym(%s) failed: %s\n", #name, dlerror());    \
    api.unload();                                                              \
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);                      \
  }

    HIP_FORALL_FUNCTIONS(LOAD_SYM)

#undef LOAD_SYM

    api.getDeviceProperties =
        reinterpret_cast<decltype(api.getDeviceProperties)>(
            dlsym(lib, "hipGetDevicePropertiesR0600"));
    if (!api.getDeviceProperties)
      api.getDeviceProperties =
          reinterpret_cast<decltype(api.getDeviceProperties)>(
              dlsym(lib, "hipGetDeviceProperties"));

    return api;
  }

  /// Close the library handle. Call only after all HIP usage is done.
  void unload() {
    if (lib) {
      dlclose(lib);
      lib = nullptr;
    }
  }
};

struct M_driver_driver;

namespace llvm {
class Twine;
} // namespace llvm

namespace M::Target::HIP {

M_driver_status_t hipErrorStatus(M_driver_driver *driver, hipError_t result,
                                 llvm::Twine const &context);

} // namespace M::Target::HIP

#undef __HIP_DISABLE_CPP_FUNCTIONS__

#endif /* HIPAPI_H */
