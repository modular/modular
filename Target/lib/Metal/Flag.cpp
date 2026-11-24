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

#include "Flag.h"
#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_api.h"

#include <new>

static NS::SharedPtr<MTL::SharedEvent> newEvent(M_driver_context &ctx) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  return NS::TransferPtr(ctx.device->mDevice->newSharedEvent());
}

DriverResult<M_driver_flag_t> M_driver_flag::create(M_driver_context &ctx) {
  NS::SharedPtr<MTL::SharedEvent> event = newEvent(ctx);
  if (!event)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  M_driver_flag_t flag = new (std::nothrow) M_driver_flag(ctx, event);
  if (flag == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return flag;
}

M_driver_status_t M_driver_flag::signal(uint64_t value) {
  std::lock_guard<std::mutex> lock(mutex);
  // Metal would silently keep the higher value; report the misuse instead.
  if (value < event->signaledValue())
    return M_DRIVER_STATUS_INVALID_ARG;
  event->setSignaledValue(value);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag::reset() {
  NS::SharedPtr<MTL::SharedEvent> fresh = newEvent(context);
  if (!fresh)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  std::lock_guard<std::mutex> lock(mutex);
  event = std::move(fresh);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_create(M_driver_context_t context,
                                       M_driver_flag_t *out_flag) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (out_flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  DriverResult<M_driver_flag_t> result = M_driver_flag::create(*context);
  if (!result)
    return result.error();
  *out_flag = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_destroy(M_driver_context_t context,
                                        M_driver_flag_t flag) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // A queue's pending wait holds its own reference to the event.
  delete flag;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_signal(M_driver_flag_t flag, uint64_t value) {
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return flag->signal(value);
}

M_driver_status_t M_driver_flag_load(M_driver_flag_t flag,
                                     uint64_t *out_value) {
  if (flag == nullptr || out_value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *out_value = flag->load();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_conditions(M_driver_device_t device,
                                           uint32_t *out_mask) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (out_mask == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // encodeWait(event, value) only tests signaledValue >= value.
  *out_mask = M_DRIVER_FLAG_CONDITION_GEQ;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_reset(M_driver_flag_t flag) {
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return flag->reset();
}
