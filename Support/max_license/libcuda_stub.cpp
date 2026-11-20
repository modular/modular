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
// On-demand CUDA driver API stub for Tracy GPU profiling and libkineto.
//
// Tracy's GPU profiling implementation (TracyCUDA.hpp) calls two CUDA driver
// API functions directly:
//
//   • cuGetErrorString     — used by the DRIVER_API_CALL error-checking macro
//   • cuStreamIsCapturing  — used to detect CUDA graph stream capture
//
// libkineto's Range.cpp wrapper (Support/lib/Profiling/Range.cpp) adds:
//
//   • cuCtxGetCurrent      — probes whether a CUDA primary context is bound
//                             to the calling thread, used to gate libkineto's
//                             prepareTrace/startTrace path (which crashes
//                             when CUPTI cannot subscribe — no live context)
//
// This shared library exports all three symbols.  On first call it opens
// libcuda.so.1 with dlopen using the same flags as MLRT's CUDADriver
// (RTLD_NOW | RTLD_GLOBAL) and resolves the real implementations.  If the
// library is not present on the machine (no CUDA driver installed, e.g. a
// CPU-only remote-execution worker) the functions return
// CUDA_ERROR_NOT_INITIALIZED, which causes Tracy to skip GPU profiling
// gracefully and libkineto's enable() to leave the CUPTI path dormant.
//
// Linking Tracy against this stub instead of directly against libcuda.so.1
// removes the hard runtime NEEDED entry for libcuda.so.1 from Tracy's DSO.
// The CUDA driver is loaded lazily and only when GPU profiling is actually
// exercised, matching the on-demand loading strategy of MLRT's CUDADriver.
//
// Note: this file intentionally avoids including <cuda.h> so that it can be
// compiled on any platform regardless of whether the CUDA SDK is installed.
// All needed types are forward-declared below, following the same pattern
// used by MLRT/lib/Driver/DeviceContext/CUDA/CUDADriver.h.
//
// License note: this stub loads libcuda.so.1 at runtime via dlopen/dlsym and
// does NOT statically link against any part of the CUDA SDK.  The CUDA SDK
// EULA restricts redistribution of statically-linked SDK code; runtime-only
// dynamic loading of the CUDA driver library (which ships with the GPU driver,
// not the SDK) is outside that scope.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Minimal CUDA driver API type forward declarations.
// These match the ABI of the real types without requiring <cuda.h>.
// ---------------------------------------------------------------------------

// CUresult: CUDA error code (32-bit int enum).
enum CUresult {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_NOT_INITIALIZED = 3,
};

// CUstream: opaque handle for a CUDA stream.
struct CUstream_st;
using CUstream = CUstream_st *;

// CUcontext: opaque handle for a CUDA context.
struct CUctx_st;
using CUcontext = CUctx_st *;

// CUstreamCaptureStatus: stream capture state enum.
enum CUstreamCaptureStatus {
  CU_STREAM_CAPTURE_STATUS_NONE = 0,
  CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
  CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
};

namespace {

// Open libcuda.so.1 on first use and cache the handle.  Returns nullptr if
// the library cannot be found (no CUDA driver installed).
//
// Uses RTLD_NOW | RTLD_GLOBAL to match the loading strategy in
// MLRT/lib/Driver/DeviceContext/CUDA/CUDADriver.cpp.  RTLD_GLOBAL makes the
// resolved symbols globally visible so that other code which expects them
// (e.g. CUPTI callbacks) can find them even without calling dlopen itself.
void *getCudaLib() {
  static void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  return lib;
}

// Resolve a single CUDA driver symbol by name from libcuda.so.1.
// Returns nullptr if the library is absent or the symbol cannot be found.
template <typename FnType>
FnType *loadSymbol(const char *name) {
  void *lib = getCudaLib();
  if (!lib)
    return nullptr;
  return reinterpret_cast<FnType *>(dlsym(lib, name));
}

} // namespace

extern "C" {

// cuGetErrorString: used by Tracy's DRIVER_API_CALL error-checking macro
// (TracyCUDA.hpp) to turn a CUresult into a human-readable string.
CUresult cuGetErrorString(CUresult error, const char **pStr) {
  using FnType = CUresult(CUresult, const char **);
  static FnType *fn = loadSymbol<FnType>("cuGetErrorString");
  if (!fn) {
    if (pStr)
      *pStr = "(CUDA driver unavailable)";
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  return fn(error, pStr);
}

// cuStreamIsCapturing: used by Tracy to detect whether a CUDA stream is
// being captured into a CUDA graph, so it can suppress profiling zones that
// would confuse the timeline (TracyCUDA.hpp).
CUresult cuStreamIsCapturing(CUstream hStream,
                             CUstreamCaptureStatus *captureStatus) {
  using FnType = CUresult(CUstream, CUstreamCaptureStatus *);
  static FnType *fn = loadSymbol<FnType>("cuStreamIsCapturing");
  if (!fn)
    return CUDA_ERROR_NOT_INITIALIZED;
  return fn(hStream, captureStatus);
}

// cuCtxGetCurrent: used by Range.cpp to test whether a CUDA primary context
// is currently bound to the calling thread.  libkineto's prepareTrace /
// startTrace subscribe to CUPTI's Activity Callback API and crash when no
// live context exists (CUPTI has nothing to subscribe to), so Range.cpp
// gates those calls on a non-null out-param here.  When libcuda.so.1 is
// absent we return CUDA_ERROR_NOT_INITIALIZED and leave *pctx untouched —
// callers treat both "error" and "*pctx == nullptr" as "no context bound."
CUresult cuCtxGetCurrent(CUcontext *pctx) {
  using FnType = CUresult(CUcontext *);
  static FnType *fn = loadSymbol<FnType>("cuCtxGetCurrent");
  if (!fn)
    return CUDA_ERROR_NOT_INITIALIZED;
  return fn(pctx);
}

// NOTE: deliberately no cuInit / cuDevicePrimaryCtxRetain / cuCtxSetCurrent
// trampolines here.  The profiling probe must never manufacture a context —
// see hasActiveCudaContext() in max/internal/profiler/lib/RangeKineto.cpp.

} // extern "C"
