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

#ifndef TARGET_LIB_METAL_EVENT_H
#define TARGET_LIB_METAL_EVENT_H

#include "Common.h"
#include "HAL/M_driver_event.h"
#include "MetalCpp.h"

#include <atomic>
#include <mutex>
#include <optional>

struct M_driver_context;

struct M_driver_event {
  // At most one is populated, based on the EVENT_FLAG_CPU_VISIBLE bit at
  // creation time.
  //
  // - sharedEvent: requested CPU-visible. Supports both host-side ops
  //   (setSignaledValue, signaledValue, waitUntilSignaledValue) and GPU-
  //   side encode ops via the MTL::Event base class.
  // - deviceEvent: intra-GPU only (no host query/wait API). Created lazily.
  NS::SharedPtr<MTL::SharedEvent> sharedEvent;
  NS::SharedPtr<MTL::Event> deviceEvent;
  MTL::Device *device = nullptr;

  // Guards deviceEvent creation, which races across queues, and the signpost.
  std::mutex mutex;
  std::optional<NS::SharedPtr<MTL::Event>> signpostTimeline;
  uint64_t signpostValue = 0; // 0: already complete

  // Latest recording's value; 0 = never recorded.
  std::atomic<uint64_t> targetValue{0};

  M_driver_event() = default;

  MTL::Event *getEvent() const {
    return sharedEvent ? static_cast<MTL::Event *>(sharedEvent.get())
                       : deviceEvent.get();
  }

  bool ensureDeviceEvent();

  [[nodiscard]] static DriverResult<M_driver_event_t>
  create(M_driver_context &ctx, M_driver_event_flags_t flags);

  static M_driver_status_t destroy(M_driver_context &ctx,
                                   M_driver_event_t event);

  static M_driver_status_t synchronize(M_driver_context &ctx,
                                       M_driver_event_t event);

  static M_driver_status_t isReady(M_driver_context &ctx,
                                   M_driver_event_t event, bool *is_ready);
};

#endif // TARGET_LIB_METAL_EVENT_H
