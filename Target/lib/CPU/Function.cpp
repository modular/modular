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
// This file implements the bundle and function entry points for the CPU HAL
// plugin. The CPU plugin does not load code bundles, so these mirror
// CpuDeviceContext::loadFunction by reporting "unimplemented".
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_function.h"

M_driver_status_t M_driver_bundle_load(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_static_bundle_t bundle,
    [[maybe_unused]] M_driver_bundle_compilation_options_t opts,
    [[maybe_unused]] M_driver_runtime_bundle_t *runtime_bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_bundle_unload([[maybe_unused]] M_driver_context_t context,
                       [[maybe_unused]] M_driver_runtime_bundle_t bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_function_load([[maybe_unused]] M_driver_context_t context,
                       [[maybe_unused]] M_driver_runtime_bundle_t bundle,
                       [[maybe_unused]] const char *function_name,
                       [[maybe_unused]] uint64_t function_name_len,
                       [[maybe_unused]] M_driver_function_t *function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_function_unload([[maybe_unused]] M_driver_context_t context,
                         [[maybe_unused]] M_driver_function_t function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_function_property([[maybe_unused]] M_driver_context_t context,
                           [[maybe_unused]] M_driver_function_t function,
                           const char *property_name, void *value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t
M_driver_function_set_attribute([[maybe_unused]] M_driver_context_t context,
                                [[maybe_unused]] M_driver_function_t function,
                                [[maybe_unused]] int32_t attribute,
                                [[maybe_unused]] int32_t value) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_function_t function,
    [[maybe_unused]] int32_t block_size,
    [[maybe_unused]] uint64_t dynamic_shared_memory_bytes,
    [[maybe_unused]] int32_t *num_blocks) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
