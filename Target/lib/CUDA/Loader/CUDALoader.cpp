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
// Runtime loading of the CUDA driver library via dlopen+dlsym.
//
//===----------------------------------------------------------------------===//

#include "CUDALoader.h"

#include <cstdio>
#include <dlfcn.h>
#include <stdlib.h>

namespace M::Target::CUDA {

DriverResult<CUDALoader> CUDALoader::load() {
  // We use both NVML and CUDA Driver to access devices. NVML discovers devices
  // in the order of PCI bus ID, so we want to force CUDA to enumerate in the
  // same order.
  // See:
  // [1] Documentation for nvmlDeviceGetHandleByIndex_v2 (non-deterministic
  //     device-ordering between reboots)
  // [2]
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);

  void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) {
    fprintf(stderr, "CUDAPlugin: dlopen(libcuda.so.1) failed: %s\n", dlerror());
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);
  }

  CUDALoader loader{};
  loader.lib = lib;

  // CUDA_XSTR(name) expands cuda.h version macros
  //
  // E.g. CUDA_FN(cuCtxCreate) where cuda.h has
  //   #define cuCtxCreate cuCtxCreate_v4:
  //
  //   CUDA_XSTR(cuCtxCreate)
  //   -> CUDA_STR_(cuCtxCreate_v4)   // arg expanded (no # yet)
  //   -> "cuCtxCreate_v4"            // # stringifies the expanded token
  //
  // A single-level #name would give "cuCtxCreate" (wrong symbol).
#define CUDA_STR_(x) #x
#define CUDA_XSTR(x) CUDA_STR_(x)

#define CUDA_FN(name)                                                          \
  loader.name =                                                                \
      reinterpret_cast<decltype(&::name)>(dlsym(lib, CUDA_XSTR(name)));        \
  if (!loader.name) {                                                          \
    fprintf(stderr, "CUDAPlugin: dlsym(%s) failed: %s\n", CUDA_XSTR(name),     \
            dlerror());                                                        \
    loader.unload();                                                           \
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);                         \
  }

#include "CUDALoader.inc"
#undef CUDA_FN
#undef CUDA_XSTR
#undef CUDA_STR_

  return loader;
}

void CUDALoader::unload() {
  if (lib) {
    dlclose(lib);
    lib = nullptr;
  }
}

} // namespace M::Target::CUDA
