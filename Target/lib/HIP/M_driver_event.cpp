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
#include "Event.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_flags_t flags,
                                        M_driver_event_t *event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (flags & ~M_DRIVER_EVENT_FLAGS_ALL)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (event == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  DriverResult<M_driver_event_t> result =
      M_driver_event::create(*context, flags);
  if (!result)
    return result.error();
  *event = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return M_driver_event::destroy(*context, event);
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return M_driver_event::synchronize(*context, event);
}

M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return M_driver_event::isReady(*context, event, is_ready);
}
