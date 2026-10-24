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

#include "Context.h"
#include "Error.h"
#include "Function.h"
#include "HAL/M_driver_api.h"

#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"
#include "llvm/ADT/StringRef.h"

M_driver_status_t
M_driver_bundle_load(M_driver_context_t context,
                     M_driver_static_bundle_t bundle,
                     M_driver_bundle_compilation_options_t opts,
                     M_driver_runtime_bundle_t *runtime_bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (runtime_bundle == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_runtime_bundle_t> result =
      M_driver_runtime_bundle::load(*context, bundle, opts);
  if (!result)
    return result.error();
  *runtime_bundle = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  return M_driver_runtime_bundle::unload(*context, bundle);
}

M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (function == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_function_t> result = M_driver_function::load(
      *context, bundle, function_name, function_name_len);
  if (!result)
    return result.error();
  *function = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return M_driver_function::unload(*context, function);
}

M_driver_status_t M_driver_function_property(M_driver_context_t context,
                                             M_driver_function_t function,
                                             const char *property_name,
                                             void *value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (llvm::StringRef(property_name) ==
      M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE) {
    M_driver_native_handle_t handle = 0;
    if (M_driver_status_t status = function->getNativeHandle(&handle);
        status != M_DRIVER_STATUS_SUCCESS)
      return status;
    *static_cast<uint64_t *>(value) = static_cast<uint64_t>(handle);
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_function_set_attribute(M_driver_context_t context,
                                                  M_driver_function_t function,
                                                  int32_t attribute,
                                                  int32_t value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return function->setAttribute(*context, attribute, value);
}

M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    M_driver_context_t context, M_driver_function_t function,
    int32_t block_size, uint64_t dynamic_shared_memory_bytes,
    int32_t *num_blocks) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return function->occupancyMaxActiveBlocks(
      *context, block_size, dynamic_shared_memory_bytes, num_blocks);
}
