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

#include "HAL/M_driver_bundle.h"
#include "Context.h"
#include "Device.h"
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_common.h"

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
