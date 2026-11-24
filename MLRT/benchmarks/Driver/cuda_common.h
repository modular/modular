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
// Common CUDA driver API utilities for C++ benchmarks.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_BENCHMARKS_ASYNCRT_CUDA_COMMON_H
#define MLRT_BENCHMARKS_ASYNCRT_CUDA_COMMON_H

#include <cstddef>
#include <dlfcn.h>

namespace CudaCommon {

// CUDA driver API types (forward declarations)
using CUcontext = struct CUctx_st *;
using CUdevice = int;
using CUdeviceptr = void *;
using CUfunction = struct CUfunc_st *;
using CUmodule = struct CUmod_st *;
using CUstream = struct CUstream_st *;
using CUresult = int;

// Launch configuration for CUDA kernel launches
struct CUlaunchConfig {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  void *attrs;
  unsigned int numAttrs;
};

// Success result for CUDA driver API calls
constexpr CUresult CUDA_SUCCESS = 0;

// Function pointer types for CUDA driver API
using CuInitFn = CUresult (*)(unsigned int);
using CuDeviceGetFn = CUresult (*)(CUdevice *, int);
using CuDevicePrimaryCtxRetainFn = CUresult (*)(CUcontext *, CUdevice);
using CuDevicePrimaryCtxReleaseFn = CUresult (*)(CUdevice);
using CuCtxSetCurrentFn = CUresult (*)(CUcontext);
using CuCtxGetCurrentFn = CUresult (*)(CUcontext *);
using CuModuleLoadDataExFn = CUresult (*)(CUmodule *, const void *,
                                          unsigned int, void *, void **);
using CuModuleUnloadFn = CUresult (*)(CUmodule);
using CuModuleGetFunctionFn = CUresult (*)(CUfunction *, CUmodule,
                                           const char *);
using CuStreamCreateFn = CUresult (*)(CUstream *, unsigned int);
using CuStreamDestroyFn = CUresult (*)(CUstream);
using CuStreamSynchronizeFn = CUresult (*)(CUstream);
using CuDeviceGetNameFn = CUresult (*)(char *, int, CUdevice);
using CuMemAllocFn = CUresult (*)(CUdeviceptr *, size_t);
using CuMemFreeFn = CUresult (*)(CUdeviceptr);
using CuLaunchKernelExFn = CUresult (*)(const CUlaunchConfig *, CUfunction,
                                        void **, void **);
using CuGetErrorStringFn = CUresult (*)(CUresult, const char **);

namespace {

// Function pointers loaded from libcuda.so
CuInitFn cuInit = nullptr;
CuDeviceGetFn cuDeviceGet = nullptr;
CuDevicePrimaryCtxRetainFn cuDevicePrimaryCtxRetain = nullptr;
CuDevicePrimaryCtxReleaseFn cuDevicePrimaryCtxRelease = nullptr;
CuCtxSetCurrentFn cuCtxSetCurrent = nullptr;
CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
CuModuleLoadDataExFn cuModuleLoadDataEx = nullptr;
CuModuleUnloadFn cuModuleUnload = nullptr;
CuModuleGetFunctionFn cuModuleGetFunction = nullptr;
CuStreamCreateFn cuStreamCreate = nullptr;
CuStreamDestroyFn cuStreamDestroy = nullptr;
CuStreamSynchronizeFn cuStreamSynchronize = nullptr;
CuDeviceGetNameFn cuDeviceGetName = nullptr;
CuMemAllocFn cuMemAlloc = nullptr;
CuMemFreeFn cuMemFree = nullptr;
CuLaunchKernelExFn cuLaunchKernelEx = nullptr;
CuGetErrorStringFn cuGetErrorString = nullptr;

} // namespace

/// Load CUDA driver functions from libcuda.so
[[maybe_unused]] static bool loadCUDAFunctions() {
  void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  }
  if (!handle)
    return false;

  cuInit = reinterpret_cast<CuInitFn>(dlsym(handle, "cuInit"));
  if (!cuInit)
    return false;

  cuDeviceGet = reinterpret_cast<CuDeviceGetFn>(dlsym(handle, "cuDeviceGet"));
  if (!cuDeviceGet)
    return false;

  cuDevicePrimaryCtxRetain = reinterpret_cast<CuDevicePrimaryCtxRetainFn>(
      dlsym(handle, "cuDevicePrimaryCtxRetain"));
  if (!cuDevicePrimaryCtxRetain)
    return false;

  cuDevicePrimaryCtxRelease = reinterpret_cast<CuDevicePrimaryCtxReleaseFn>(
      dlsym(handle, "cuDevicePrimaryCtxRelease_v2"));
  if (!cuDevicePrimaryCtxRelease)
    return false;

  cuCtxSetCurrent =
      reinterpret_cast<CuCtxSetCurrentFn>(dlsym(handle, "cuCtxSetCurrent"));
  if (!cuCtxSetCurrent)
    return false;

  cuCtxGetCurrent =
      reinterpret_cast<CuCtxGetCurrentFn>(dlsym(handle, "cuCtxGetCurrent"));
  if (!cuCtxGetCurrent)
    return false;

  cuModuleLoadDataEx = reinterpret_cast<CuModuleLoadDataExFn>(
      dlsym(handle, "cuModuleLoadDataEx"));
  if (!cuModuleLoadDataEx)
    return false;

  cuModuleUnload =
      reinterpret_cast<CuModuleUnloadFn>(dlsym(handle, "cuModuleUnload"));
  if (!cuModuleUnload)
    return false;

  cuModuleGetFunction = reinterpret_cast<CuModuleGetFunctionFn>(
      dlsym(handle, "cuModuleGetFunction"));
  if (!cuModuleGetFunction)
    return false;

  cuStreamCreate =
      reinterpret_cast<CuStreamCreateFn>(dlsym(handle, "cuStreamCreate"));
  if (!cuStreamCreate)
    return false;

  cuStreamDestroy =
      reinterpret_cast<CuStreamDestroyFn>(dlsym(handle, "cuStreamDestroy"));
  if (!cuStreamDestroy)
    return false;

  cuStreamSynchronize = reinterpret_cast<CuStreamSynchronizeFn>(
      dlsym(handle, "cuStreamSynchronize"));
  if (!cuStreamSynchronize)
    return false;

  cuDeviceGetName =
      reinterpret_cast<CuDeviceGetNameFn>(dlsym(handle, "cuDeviceGetName"));
  if (!cuDeviceGetName)
    return false;

  cuMemAlloc = reinterpret_cast<CuMemAllocFn>(dlsym(handle, "cuMemAlloc_v2"));
  if (!cuMemAlloc)
    return false;

  cuMemFree = reinterpret_cast<CuMemFreeFn>(dlsym(handle, "cuMemFree_v2"));
  if (!cuMemFree)
    return false;

  cuLaunchKernelEx =
      reinterpret_cast<CuLaunchKernelExFn>(dlsym(handle, "cuLaunchKernelEx"));
  if (!cuLaunchKernelEx)
    return false;

  cuGetErrorString =
      reinterpret_cast<CuGetErrorStringFn>(dlsym(handle, "cuGetErrorString"));
  if (!cuGetErrorString)
    return false;

  return true;
}

} // namespace CudaCommon

#endif // MLRT_BENCHMARKS_ASYNCRT_CUDA_COMMON_H
