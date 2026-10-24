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
#include "HAL/M_driver_api.h"

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

DriverResult<M_driver_event_t>
M_driver_event::create(M_driver_context &ctx, M_driver_event_flags_t flags) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  M_driver_event_t event = new (std::nothrow) M_driver_event();
  if (event == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  if (flags & M_DRIVER_EVENT_FLAG_CPU_VISIBLE) {
    NS::SharedPtr<MTL::SharedEvent> shared =
        NS::TransferPtr(ctx.device->mDevice->newSharedEvent());
    if (!shared) {
      delete event;
      return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
    }
    event->sharedEvent = std::move(shared);
  } else {
    NS::SharedPtr<MTL::Event> plain =
        NS::TransferPtr(ctx.device->mDevice->newEvent());
    if (!plain) {
      delete event;
      return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
    }
    event->deviceEvent = std::move(plain);
  }

  return event;
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  if (!event->sharedEvent)
    return M_DRIVER_STATUS_INVALID_ARG;
  uint64_t target = event->targetValue.load(std::memory_order_acquire);
  if (target == 0)
    return M_DRIVER_STATUS_SUCCESS;
  event->sharedEvent->waitUntilSignaledValue(target, UINT64_MAX);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (!event->sharedEvent)
    return M_DRIVER_STATUS_INVALID_ARG;
  uint64_t target = event->targetValue.load(std::memory_order_acquire);
  *is_ready = (target == 0 || event->sharedEvent->signaledValue() >= target);
  return M_DRIVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// C API dispatch
// ---------------------------------------------------------------------------

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
