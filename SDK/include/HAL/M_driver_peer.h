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

#ifndef M_DRIVER_PEER_H
#define M_DRIVER_PEER_H

#include "M_driver_common.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Report whether kernels on `device` may dereference memory that `peer` owns
 * once peer access is enabled. A device is not its own peer.
 *
 * Optional extension: a plugin exports both peer symbols or neither, and
 * callers treat absence as "no peers".
 *
 * @param[in] device The device that would read or write.
 * @param[in] peer The device whose memory would be accessed.
 * @param[out] out_can_access Set to true if the pair supports peer access.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_DEVICE if either
 *         device is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_can_access_peer(M_driver_device_t device,
                                                  M_driver_device_t peer,
                                                  bool *out_can_access);

typedef M_driver_status_t (*M_driver_device_can_access_peer_func_ptr_t)(
    M_driver_device_t device, M_driver_device_t peer, bool *out_can_access);

/**
 * Make every allocation that `peer` owns, past and future, addressable by
 * kernels on `context`. Idempotent. Both contexts must come from the same
 * driver instance.
 *
 * @param[in] context The context whose kernels gain access.
 * @param[in] peer The context whose memory becomes accessible.
 * @return Standard Status Protocol, M_DRIVER_STATUS_BAD_CONTEXT if either
 *         context is not valid, or a plugin error if the hardware pair does
 *         not support peer access.
 */
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_context_enable_peer_access(M_driver_context_t context,
                                    M_driver_context_t peer);

typedef M_driver_status_t (*M_driver_context_enable_peer_access_func_ptr_t)(
    M_driver_context_t context, M_driver_context_t peer);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_PEER_H
