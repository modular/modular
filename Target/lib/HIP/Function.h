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

#ifndef TARGET_LIB_HIP_FUNCTION_H
#define TARGET_LIB_HIP_FUNCTION_H

#include "Error.h"
#include "HAL/M_driver_bundle.h"
#include "HIPApi.h"

struct M_driver_context;

struct M_driver_runtime_bundle {
  hipModule_t module;

  [[nodiscard]] static DriverResult<M_driver_runtime_bundle_t>
  load(M_driver_context &ctx, M_driver_static_bundle_t bundle,
       M_driver_bundle_compilation_options_t opts);

  static M_driver_status_t unload(M_driver_context &ctx,
                                  M_driver_runtime_bundle_t bundle);
};

struct M_driver_function {
  hipFunction_t function;

  [[nodiscard]] static DriverResult<M_driver_function_t>
  load(M_driver_context &ctx, M_driver_runtime_bundle_t bundle,
       const char *function_name, uint64_t function_name_len);

  static M_driver_status_t unload(M_driver_function_t func);

  M_driver_status_t
  occupancyMaxActiveBlocks(M_driver_context &ctx, int32_t block_size,
                           uint64_t dynamic_shared_memory_bytes,
                           int32_t *num_blocks);
};

#endif // TARGET_LIB_HIP_FUNCTION_H
