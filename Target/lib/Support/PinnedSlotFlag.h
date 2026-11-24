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

#ifndef TARGET_LIB_SUPPORT_PINNED_SLOT_FLAG_H
#define TARGET_LIB_SUPPORT_PINNED_SLOT_FLAG_H

#include "HAL/M_driver_flag.h"
#include "HAL/M_driver_memory.h"

#include <atomic>
#include <cstdint>

struct M_driver_context;

/// The CUDA and HIP completion flag: a pinned, device-mapped 8-byte slot whose
/// host pointer, under unified addressing, is the device address the stream
/// polls. Each plugin defines its entry points with PinnedSlotFlag.inc.
struct M_driver_flag {
  // The stream polls the slot as a plain 64-bit word.
  static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t) &&
                std::atomic<uint64_t>::is_always_lock_free);

  M_driver_context &context;
  M_driver_memory_t memory;
  std::atomic<uint64_t> *slot;

  uint64_t deviceAddress() const { return reinterpret_cast<uint64_t>(slot); }

  void signal(uint64_t value) {
    // Release pairs with the stream wait's acquire and with load().
    slot->store(value, std::memory_order_release);
  }

  uint64_t load() const { return slot->load(std::memory_order_acquire); }

  void reset() {
    // Reallocating would synchronize the device and deadlock against a wait
    // queued for the next signal.
    slot->store(0, std::memory_order_relaxed);
  }
};

#endif // TARGET_LIB_SUPPORT_PINNED_SLOT_FLAG_H
