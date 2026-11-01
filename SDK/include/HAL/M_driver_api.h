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
#ifndef M_DRIVER_API_H
#define M_DRIVER_API_H

#include "M_driver_bundle.h"
#include "M_driver_common.h"
#include "M_driver_context.h"
#include "M_driver_device.h"
#include "M_driver_driver.h"
#include "M_driver_event.h"
#include "M_driver_function.h"
#include "M_driver_memory.h"
#include "M_driver_peer.h"
#include "M_driver_queue.h"
#include "M_driver_tma.h"

/**
 * Standard status protocol
 * -----------------------
 * All API-functions MUST return one of the following status codes or one
 * from their respective official descriptions.
 * - M_DRIVER_STATUS_SUCCESS if the operation was successful.
 * - An implementation-defined error code whose human-readable description can
 *   be retrieved by providing it to M_driver_status_message().
 * - M_DRIVER_STATUS_UNKNOWN_ERROR in the worst case where it is impossible to
    determine the actual error.
 */

#endif /* M_DRIVER_API_H */
