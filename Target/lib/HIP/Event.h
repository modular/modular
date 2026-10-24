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

#ifndef TARGET_LIB_HIP_EVENT_H
#define TARGET_LIB_HIP_EVENT_H

#include "HAL/M_driver_event.h"
#include "HIPApi.h"

struct M_driver_context;

struct M_driver_event {
  hipEvent_t event;

  explicit M_driver_event(hipEvent_t event) : event(event) {}

  [[nodiscard]] static DriverResult<M_driver_event_t>
  create(M_driver_context &ctx, M_driver_event_flags_t flags);

  static M_driver_status_t destroy(M_driver_context &ctx,
                                   M_driver_event_t event);

  static M_driver_status_t synchronize(M_driver_context &ctx,
                                       M_driver_event_t event);

  static M_driver_status_t isReady(M_driver_context &ctx,
                                   M_driver_event_t event, bool *is_ready);
};

#endif // TARGET_LIB_HIP_EVENT_H
