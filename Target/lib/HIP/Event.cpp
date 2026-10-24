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

#include "Event.h"
#include "Context.h"
#include "Device.h"

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_event_t>
M_driver_event::create(M_driver_context &ctx, M_driver_event_flags_t flags) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  // HIP events are always host-queryable, so CPU_VISIBLE
  // and currently has no effect.
  (void)flags;

  HIPApi &hip = dev.getDriver().hip();
  hipEvent_t event;
  if (hipError_t r = hip.hipEventCreateWithFlags(&event, hipEventDisableTiming);
      r != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&dev.getDriver(), r, "hipEventCreateWithFlags"));

  return new M_driver_event(event);
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipEventDestroy(event->event); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipEventDestroy");

  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipEventSynchronize(event->event); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipEventSynchronize");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipEventQuery(event->event);
  if (result == hipSuccess) {
    *is_ready = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (result == hipErrorNotReady) {
    *is_ready = false;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return hipErrorStatus(&dev.getDriver(), result, "hipEventQuery");
}
