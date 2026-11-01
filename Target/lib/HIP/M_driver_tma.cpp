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

// HIP has no TMA engine, so the descriptor encoders report
// M_DRIVER_STATUS_INCOMPATIBLE. The symbols must still exist because plugin
// entry points are resolved eagerly at load time.

#include "HAL/M_driver_api.h"

M_driver_status_t
M_driver_tensor_map_encode_tiled(M_driver_context_t, void *, int32_t, uint32_t,
                                 uint64_t, const uint64_t *, const uint64_t *,
                                 const uint32_t *, const uint32_t *, int32_t,
                                 int32_t, int32_t, int32_t) {
  return M_DRIVER_STATUS_INCOMPATIBLE;
}

M_driver_status_t M_driver_tensor_map_encode_im2col(
    M_driver_context_t, void *, int32_t, uint32_t, uint64_t, const uint64_t *,
    const uint64_t *, const int32_t *, const int32_t *, int32_t, int32_t,
    const uint32_t *, int32_t, int32_t, int32_t, int32_t) {
  return M_DRIVER_STATUS_INCOMPATIBLE;
}
