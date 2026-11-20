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

#ifndef TARGET_LIB_METAL_FLAG_H
#define TARGET_LIB_METAL_FLAG_H

#include "Common.h"
#include "HAL/M_driver_flag.h"
#include "MetalCpp.h"

#include <mutex>

struct M_driver_context;

struct M_driver_flag {
  M_driver_context &context;
  // Guards `event`, which reset replaces because an MTLSharedEvent cannot
  // decrease, and makes signal's no-decrease check and store atomic.
  std::mutex mutex;
  NS::SharedPtr<MTL::SharedEvent> event;

  M_driver_flag(M_driver_context &context,
                NS::SharedPtr<MTL::SharedEvent> event)
      : context(context), event(std::move(event)) {}

  [[nodiscard]] static DriverResult<M_driver_flag_t>
  create(M_driver_context &ctx);

  NS::SharedPtr<MTL::SharedEvent> currentEvent() {
    std::lock_guard<std::mutex> lock(mutex);
    return event;
  }

  M_driver_status_t signal(uint64_t value);
  uint64_t load() { return currentEvent()->signaledValue(); }
  M_driver_status_t reset();
};

#endif // TARGET_LIB_METAL_FLAG_H
