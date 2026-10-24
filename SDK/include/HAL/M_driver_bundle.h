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

#ifndef HAL_M_DRIVER_BUNDLE_H
#define HAL_M_DRIVER_BUNDLE_H

#include "M_driver_common.h"
#include <stddef.h>

#ifdef __cplusplus
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// A static bundle intuitively corresponds to a code object file,
/// module, or library. It describes its type, and contains a non-owning
/// reference to the data backing it, but does not have additional
/// ownership semantics and should be a format that is acquirable
/// without talking to the driver first.
///
/// It is transparently represented here such that both plugins
/// and HAL can construct and read it.
///
/// It does not imply or allow any related memory footprint on
/// devices until such time as it is turned into a
/// M_driver_runtime_bundle by loading it.
struct M_driver_static_bundle {
  /// A reference to the backing data for this bundle
  struct M_driver_slice mapped_data;

  /// A plugin-specific file type annotation for
  /// the format behind `mapped_data`, to use for
  /// disambiguation between plugins
  /// (a static bundle from one is attempted to be used against
  /// a different bundle), or for a plugin to support multiple
  /// distinct file formats as input.
  const char *file_type;
  uint64_t file_type_len;
};

typedef struct M_driver_static_bundle *M_driver_static_bundle_t;

/// A collection of load- or compile- time inputs to accompany
/// a static bundle.
///
/// TODO(Sawyer): assess alternatives for this that are
/// plugin agnostic, or at least more extensible.
struct M_driver_bundle_compilation_options {
  /// Null-terminated UTF-8 string describing
  /// debug profile to use. Values are plugin specific.
  const char *debug_level;
  uint64_t debug_level_len;

  /// Scale is plugin specific, usability plugin specific.
  int32_t optimization_level;
};

typedef struct M_driver_bundle_compilation_options
    *M_driver_bundle_compilation_options_t;

/// A runtime bundle is returned by a plugin as a result of loading a
/// static bundle. A runtime bundle is tightly bound to each plugin,
/// and is not portable or translatable between plugins or drivers.
///
/// It owns any runtime information and allocations that are related to
/// holding a loaded module or library in memory, and is allowed to have
/// some cost in terms of device memory to keep loaded.
struct M_driver_runtime_bundle;

typedef struct M_driver_runtime_bundle *M_driver_runtime_bundle_t;

/// Use to load a code object containing a module/library or
/// some other bundle of target-specific executable code,
/// to construct a runtime bundle from a static bundle.
/// @param[in] context The context to load the bundle into.
/// @param[in] bundle The bundle to register.
/// @param[out] runtime_bundle The runtime bundle.
/// @return M_DRIVER_STATUS_SUCCESS, or:
/// - M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
/// - M_DRIVER_STATUS_BAD_BUNDLE if the bundle ptr is not valid.
/// - M_DRIVER_STATUS_INCOMPATIBLE if the file_type for this bundle is
/// not supported by this plugin.
/// - M_DRIVER_STATUS_UNKNOWN_ERROR if an unknown error occurred.
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_bundle_load(M_driver_context_t context,
                     M_driver_static_bundle_t bundle,
                     M_driver_bundle_compilation_options_t opts,
                     M_driver_runtime_bundle_t *runtime_bundle);

typedef M_driver_status_t (*M_driver_bundle_load_func_ptr_t)(
    M_driver_context_t context, M_driver_static_bundle_t bundle,
    M_driver_bundle_compilation_options_t opts,
    M_driver_runtime_bundle_t *runtime_bundle);

/// Use to unload a previously loaded bundle and release associated resources.
/// @param[in] context The context the bundle was loaded into.
/// @param[in] bundle The bundle to unload.
/// @return M_DRIVER_STATUS_SUCCESS, or:
/// - M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
/// - M_DRIVER_STATUS_BAD_BUNDLE if the bundle ptr is not valid.
/// - M_DRIVER_STATUS_UNKNOWN_ERROR if an unknown error occurred.
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle);

typedef M_driver_status_t (*M_driver_bundle_unload_func_ptr_t)(
    M_driver_context_t context, M_driver_runtime_bundle_t bundle);

#ifdef __cplusplus
}
#endif

#endif // HAL_M_DRIVER_BUNDLE_H
