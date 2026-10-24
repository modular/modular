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
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"

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
  return M_driver_function::unload(function);
}

// Mirrors the HIP DeviceContext: it exposes no native module handle and does
// not support setting function attributes on module-loaded functions.

M_driver_status_t
M_driver_function_property([[maybe_unused]] M_driver_context_t context,
                           [[maybe_unused]] M_driver_function_t function,
                           const char *property_name, void *value) {
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t
M_driver_function_set_attribute([[maybe_unused]] M_driver_context_t context,
                                [[maybe_unused]] M_driver_function_t function,
                                [[maybe_unused]] int32_t attribute,
                                [[maybe_unused]] int32_t value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
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
