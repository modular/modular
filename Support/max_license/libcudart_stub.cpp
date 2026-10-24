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
// On-demand CUDA runtime API stub for GPU profiling.
//
// Tracy's GPU profiling implementation (TracyCUDA.hpp) calls
// cudaDeviceSynchronize directly.  The libkineto profiler link-references a
// handful of additional CUDA runtime symbols (version/device query,
// error-string lookup, kernel launch interception).
//
// This shared library exports every symbol both consumers need.  On first
// call each trampoline opens libcudart.so with dlopen (RTLD_NOW | RTLD_GLOBAL)
// and resolves the real implementation via dlsym, caching the function
// pointer in a function-scope static.  If the library is not present on the
// machine (no CUDA toolkit installed, e.g. a CPU-only remote-execution
// worker) each trampoline returns cudaErrorInsufficientDriver so the consumer
// gracefully skips GPU profiling.  cudaGetErrorString is special-cased to
// return a placeholder string instead of an error code.
//
// Linking against this stub instead of directly against libcudart.so loads the
// CUDA runtime lazily, only when GPU profiling is actually exercised.  The
// companion libcuda_stub and libcupti_stub stubs follow the same dlopen/dlsym
// approach; new stub additions should use the macro shape below.
//
// License note: this stub loads libcudart.so at runtime via dlopen/dlsym and
// does NOT statically link against any part of the CUDA SDK.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include <cuda_runtime_api.h>

namespace {

// Open libcudart.so on first use and cache the handle.  Returns nullptr if
// the library cannot be found (no CUDA toolkit installed).
//
// Uses RTLD_NOW | RTLD_GLOBAL to match the loading strategy in the companion
// libcuda_stub and libcupti_stub stubs.
void *getCudartLib() {
  static void *lib = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  return lib;
}

} // namespace

// Two-level stringize so the dlsym lookup uses the *expanded* token, keeping it
// in sync with the symbol that the function definition and `decltype(&::NAME)`
// resolve to. The currently pinned cuda_x86_64 redistributable does not
// version-alias its runtime entry points, so this is a no-op today. But some
// CUDA toolkit versions alias a name to a versioned symbol (e.g.
// `cudaGetDeviceProperties` -> `cudaGetDeviceProperties_v2`); were that to
// arrive with a header bump, a single-level `#NAME` would stringize the
// pre-expansion token and resolve the wrong symbol. This keeps the stub correct
// across header changes at zero cost for non-aliased names.
#define KINETO_STUB_STR(x) #x
#define KINETO_STUB_XSTR(x) KINETO_STUB_STR(x)

// Each trampoline: dlsym the real symbol on first call (cached via static), and
// return cudaErrorInsufficientDriver if libcudart.so is missing.  Invoke it as
// a complete statement, then forward to the cached pointer:
//
//   CUDART_STUB_TRAMPOLINE(cudaFoo);
//   return cudaFoo_fn(args);
#define CUDART_STUB_TRAMPOLINE(NAME)                                           \
  using NAME##_FnType = decltype(&::NAME);                                     \
  static NAME##_FnType NAME##_fn = []()->NAME##_FnType {                       \
    void *lib = getCudartLib();                                                \
    return lib ? reinterpret_cast<NAME##_FnType>(                              \
                     dlsym(lib, KINETO_STUB_XSTR(NAME)))                       \
               : nullptr;                                                      \
  }                                                                            \
  ();                                                                          \
  if (!NAME##_fn)                                                              \
    return cudaErrorInsufficientDriver;

extern "C" {

// cudaDeviceSynchronize: used by Tracy's BeginInstrumentation() and
// EndInstrumentation() (TracyCUDA.hpp) to ensure all device work completes
// before starting or stopping GPU profiling.
cudaError_t cudaDeviceSynchronize() {
  CUDART_STUB_TRAMPOLINE(cudaDeviceSynchronize);
  return cudaDeviceSynchronize_fn();
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
  CUDART_STUB_TRAMPOLINE(cudaDriverGetVersion);
  return cudaDriverGetVersion_fn(driverVersion);
}

cudaError_t cudaGetDeviceCount(int *count) {
  CUDART_STUB_TRAMPOLINE(cudaGetDeviceCount);
  return cudaGetDeviceCount_fn(count);
}

// Leaves *prop unchanged when libcudart.so is absent; callers must check the
// return code before reading prop.
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  CUDART_STUB_TRAMPOLINE(cudaGetDeviceProperties);
  return cudaGetDeviceProperties_fn(prop, device);
}

// cudaGetErrorString: returns const char *, not cudaError_t.  Special-cased so
// callers always get a non-null string even when libcudart.so is unavailable.
const char *cudaGetErrorString(cudaError_t error) {
  using FnType = decltype(&::cudaGetErrorString);
  static FnType fn = []() -> FnType {
    void *lib = getCudartLib();
    return lib ? reinterpret_cast<FnType>(
                     dlsym(lib, KINETO_STUB_XSTR(cudaGetErrorString)))
               : nullptr;
  }();
  if (!fn)
    return "(libcudart.so unavailable)";
  return fn(error);
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                const void *func, void **args) {
  CUDART_STUB_TRAMPOLINE(cudaLaunchKernelExC);
  return cudaLaunchKernelExC_fn(config, func, args);
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
  CUDART_STUB_TRAMPOLINE(cudaRuntimeGetVersion);
  return cudaRuntimeGetVersion_fn(runtimeVersion);
}

} // extern "C"

#undef CUDART_STUB_TRAMPOLINE
#undef KINETO_STUB_XSTR
#undef KINETO_STUB_STR
