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

// The completion-flag entry points shared by the CUDA and HIP plugins. The
// including file first includes its plugin's Context.h, Memory.h and Queue.h,
// which supply M_driver_context::allocPinned/freePinned, M_driver_memory and
// M_driver_queue::{kFlagConditions,waitFlag}.

#include "HAL/M_driver_api.h"
#include "Target/lib/Support/PinnedSlotFlag.h"

#include <new>

M_driver_status_t M_driver_flag_create(M_driver_context_t context,
                                       M_driver_flag_t *out_flag) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (out_flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  DriverResult<M_driver_memory_t> memory =
      context->allocPinned(sizeof(uint64_t));
  if (!memory)
    return memory.error();
  auto *slot = new ((*memory)->ptr) std::atomic<uint64_t>(0);
  M_driver_flag_t flag =
      new (std::nothrow) M_driver_flag{*context, *memory, slot};
  if (flag == nullptr) {
    (void)context->freePinned(*memory);
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  }
  *out_flag = flag;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_destroy(M_driver_context_t context,
                                        M_driver_flag_t flag) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // Freeing synchronizes the device, so the API forbids a pending wait.
  M_driver_status_t status = flag->context.freePinned(flag->memory);
  delete flag;
  return status;
}

M_driver_status_t M_driver_flag_signal(M_driver_flag_t flag, uint64_t value) {
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  flag->signal(value);
  return M_DRIVER_STATUS_SUCCESS;
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
  *out_mask = M_driver_queue::kFlagConditions;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue_wait_flag(M_driver_queue_t queue, M_driver_flag_t flag,
                         uint64_t value, M_driver_flag_condition_t condition) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return queue->waitFlag(flag, value, condition);
}

M_driver_status_t M_driver_flag_reset(M_driver_flag_t flag) {
  if (flag == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  flag->reset();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_flag_device_address(M_driver_flag_t flag,
                                               uint64_t *out_address) {
  if (flag == nullptr || out_address == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *out_address = flag->deviceAddress();
  return M_DRIVER_STATUS_SUCCESS;
}
