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
// On-demand rocprofiler-sdk stub for Tracy GPU profiling.
//
// Tracy's ROCm GPU profiling implementation (TracyRocprof.cpp) calls the
// following rocprofiler-sdk API functions directly:
//
//   * rocprofiler_get_status_string                          — error checking
//   * rocprofiler_create_context                             — context creation
//   * rocprofiler_configure_callback_dispatch_counting_service — counter setup
//   * rocprofiler_configure_callback_tracing_service          — tracing setup
//   * rocprofiler_start_context                              — start profiling
//   * rocprofiler_stop_context                               — stop profiling
//   * rocprofiler_iterate_agent_supported_counters           — counter
//   enumeration
//   * rocprofiler_query_counter_info                         — counter metadata
//   * rocprofiler_query_record_counter_id                    — counter ID query
//   * rocprofiler_create_profile_config                      — profile config
//   * rocprofiler_create_counter_config                      — counter config
//     (new name for create_profile_config, called by deprecated inline wrapper)
//
// This shared library exports all eleven symbols.  On first call it opens
// librocprofiler-sdk.so with dlopen (RTLD_NOW | RTLD_GLOBAL) and resolves the
// real implementations.  If the library is not present on the machine (no ROCm
// SDK / AMD GPU, e.g. a CPU-only or NVIDIA-only worker) each function returns
// ROCPROFILER_STATUS_ERROR, which causes Tracy to skip ROCm GPU profiling
// gracefully.
//
// Linking Tracy against this stub instead of directly against
// librocprofiler-sdk.so removes the hard runtime NEEDED entry for
// librocprofiler-sdk.so from Tracy's DSO.  The library is loaded lazily and
// only when GPU profiling is actually exercised, matching the on-demand
// loading strategy used by the CUDA/CUPTI stubs.
//
// Note: this file intentionally avoids including
// <rocprofiler-sdk/rocprofiler.h> so that it can be compiled on any platform
// regardless of whether the ROCm SDK is installed.  All needed types are
// forward-declared below, following the same pattern used by the CUDA/CUPTI
// stubs.
//
// The extern "C" function rocprofiler_configure() is NOT stubbed here because
// it is EXPORTED by Tracy's TracyRocprof.cpp (not called into rocprofiler-sdk).
// The rocprofiler runtime discovers and calls that symbol to register Tracy as
// a profiling tool.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Minimal rocprofiler-sdk type forward declarations.
// These match the ABI of the real types without requiring
// <rocprofiler-sdk/rocprofiler.h>.
// ---------------------------------------------------------------------------

// rocprofiler_status_t: status/error code enum.
enum rocprofiler_status_t {
  ROCPROFILER_STATUS_SUCCESS = 0,
  ROCPROFILER_STATUS_ERROR = 1,
};

// Opaque handle types used by rocprofiler-sdk.
// These are all simple structs containing a uint64_t handle.
struct rocprofiler_context_id_t {
  uint64_t handle;
};

struct rocprofiler_agent_id_t {
  uint64_t handle;
};

struct rocprofiler_counter_id_t {
  uint64_t handle;
};

struct rocprofiler_profile_config_id_t {
  uint64_t handle;
};

// rocprofiler_counter_config_id_t is the new name for profile_config_id_t.
// The deprecated rocprofiler_create_profile_config inline function calls
// rocprofiler_create_counter_config which uses this type.
using rocprofiler_counter_config_id_t = rocprofiler_profile_config_id_t;

// Callback tracing types.
enum rocprofiler_callback_tracing_kind_t {
  ROCPROFILER_CALLBACK_TRACING_NONE = 0
};
using rocprofiler_tracing_operation_t = uint32_t;

// Counter info version enum.
enum rocprofiler_counter_info_version_id_t {
  ROCPROFILER_COUNTER_INFO_VERSION_0 = 0
};

// Record counter ID type.
struct rocprofiler_counter_instance_id_t {
  uint64_t handle;
};

// Note: callback function pointer types are not forward-declared here because
// the stub functions accept them as opaque void* parameters and pass them
// through to the real library.  This avoids having to replicate the full
// rocprofiler-sdk callback type definitions.

namespace {

// Open librocprofiler-sdk.so on first use and cache the handle.  Returns
// nullptr if the library cannot be found (no ROCm SDK installed).
void *getRocprofilerLib() {
  static void *lib = dlopen("librocprofiler-sdk.so", RTLD_NOW | RTLD_GLOBAL);
  return lib;
}

// Resolve a single rocprofiler-sdk symbol by name.
// Returns nullptr if the library is absent or the symbol cannot be found.
template <typename FnType>
FnType *loadSymbol(const char *name) {
  void *lib = getRocprofilerLib();
  if (!lib)
    return nullptr;
  return reinterpret_cast<FnType *>(dlsym(lib, name));
}

} // namespace

extern "C" {

// rocprofiler_get_status_string: used by Tracy's ROCPROFILER_CALL error-
// checking macro to convert a status code into a human-readable string.
const char *rocprofiler_get_status_string(rocprofiler_status_t status) {
  using FnType = const char *(rocprofiler_status_t);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_get_status_string");
  if (!fn)
    return "(rocprofiler-sdk unavailable)";
  return fn(status);
}

// rocprofiler_create_context: creates a profiling context.
rocprofiler_status_t
rocprofiler_create_context(rocprofiler_context_id_t *context_id) {
  using FnType = rocprofiler_status_t(rocprofiler_context_id_t *);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_create_context");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(context_id);
}

// rocprofiler_configure_callback_dispatch_counting_service: sets up hardware
// counter collection for kernel dispatches.
// Callback function pointers are passed as void* to avoid replicating the
// full rocprofiler-sdk callback type definitions.
rocprofiler_status_t rocprofiler_configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t context_id, void *dispatch_callback,
    void *dispatch_callback_data, void *record_callback,
    void *record_callback_data) {
  using FnType = rocprofiler_status_t(rocprofiler_context_id_t, void *, void *,
                                      void *, void *);
  static FnType *fn = loadSymbol<FnType>(
      "rocprofiler_configure_callback_dispatch_counting_service");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(context_id, dispatch_callback, dispatch_callback_data,
            record_callback, record_callback_data);
}

// rocprofiler_configure_callback_tracing_service: configures callback-based
// tracing for code object, kernel dispatch, or memory copy events.
// The callback function pointer is passed as void* to avoid replicating the
// full rocprofiler-sdk callback type definitions.
rocprofiler_status_t rocprofiler_configure_callback_tracing_service(
    rocprofiler_context_id_t context_id,
    rocprofiler_callback_tracing_kind_t kind,
    const rocprofiler_tracing_operation_t *operations, size_t operations_count,
    void *callback, void *callback_data) {
  using FnType = rocprofiler_status_t(
      rocprofiler_context_id_t, rocprofiler_callback_tracing_kind_t,
      const rocprofiler_tracing_operation_t *, size_t, void *, void *);
  static FnType *fn =
      loadSymbol<FnType>("rocprofiler_configure_callback_tracing_service");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(context_id, kind, operations, operations_count, callback,
            callback_data);
}

// rocprofiler_start_context: begins profiling on a context.
rocprofiler_status_t
rocprofiler_start_context(rocprofiler_context_id_t context_id) {
  using FnType = rocprofiler_status_t(rocprofiler_context_id_t);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_start_context");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(context_id);
}

// rocprofiler_stop_context: stops profiling on a context.
rocprofiler_status_t
rocprofiler_stop_context(rocprofiler_context_id_t context_id) {
  using FnType = rocprofiler_status_t(rocprofiler_context_id_t);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_stop_context");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(context_id);
}

// rocprofiler_iterate_agent_supported_counters: iterates over the hardware
// counters supported by a specific GPU agent.
// The callback function pointer is passed as void* to avoid replicating the
// full rocprofiler-sdk callback type definitions.
rocprofiler_status_t
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t agent_id,
                                             void *callback, void *user_data) {
  using FnType = rocprofiler_status_t(rocprofiler_agent_id_t, void *, void *);
  static FnType *fn =
      loadSymbol<FnType>("rocprofiler_iterate_agent_supported_counters");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(agent_id, callback, user_data);
}

// rocprofiler_query_counter_info: retrieves metadata about a hardware counter.
rocprofiler_status_t
rocprofiler_query_counter_info(rocprofiler_counter_id_t counter_id,
                               rocprofiler_counter_info_version_id_t version,
                               void *info) {
  using FnType = rocprofiler_status_t(
      rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t, void *);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_query_counter_info");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(counter_id, version, info);
}

// rocprofiler_query_record_counter_id: extracts the counter ID from a record.
rocprofiler_status_t
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t record_id,
                                    rocprofiler_counter_id_t *counter_id) {
  using FnType = rocprofiler_status_t(rocprofiler_counter_instance_id_t,
                                      rocprofiler_counter_id_t *);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_query_record_counter_id");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(record_id, counter_id);
}

// rocprofiler_create_profile_config: creates a profile configuration for
// collecting a set of hardware counters on a specific agent.
rocprofiler_status_t rocprofiler_create_profile_config(
    rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t *counters_list,
    size_t counters_count, rocprofiler_profile_config_id_t *config_id) {
  using FnType =
      rocprofiler_status_t(rocprofiler_agent_id_t, rocprofiler_counter_id_t *,
                           size_t, rocprofiler_profile_config_id_t *);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_create_profile_config");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(agent_id, counters_list, counters_count, config_id);
}

// rocprofiler_create_counter_config: new name for create_profile_config.
// The deprecated inline rocprofiler_create_profile_config() in
// <rocprofiler-sdk/deprecated/profile_config.h> calls this symbol.
rocprofiler_status_t rocprofiler_create_counter_config(
    rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t *counters_list,
    size_t counters_count, rocprofiler_counter_config_id_t *config_id) {
  using FnType =
      rocprofiler_status_t(rocprofiler_agent_id_t, rocprofiler_counter_id_t *,
                           size_t, rocprofiler_counter_config_id_t *);
  static FnType *fn = loadSymbol<FnType>("rocprofiler_create_counter_config");
  if (!fn)
    return ROCPROFILER_STATUS_ERROR;
  return fn(agent_id, counters_list, counters_count, config_id);
}

} // extern "C"
