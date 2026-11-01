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

#ifndef MLRT_CORE_TENSOR_STATECONTEXT_H
#define MLRT_CORE_TENSOR_STATECONTEXT_H

#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/StateSlot.h"

#include <mutex>
#include <vector>

namespace M::MLRT {

/// A state context holds a number of 'state slots'. The number of state slots
/// is set at construction time and cannot be changed. Each state slot can hold
/// an arbitrary, reference counted payload (which should derive from
/// ReferenceCounted). Generally the state context is created during a model's
/// 'init' phase, each slot is setup using unique slot indexes determined at
/// compile time, and the state context is reused by each call to the model's
/// 'execute' phase.
///
/// State slots may hold state context for specific operations in the model, or
/// may hold framework or runtime specific state shared across many operations
/// in the model. State slots also support generically caching buffer
/// allocations.
///
/// No thread synchronization is enforced. A slot index should be set or
/// emplaced at most once, which is easy to enforce if the slots are setup
/// only in the 'init' phase. A state slot's contents should never be
/// concurrently mutated and/or read, which is easy to enforce if the slot
/// is used by exactly one operation.
///
/// Note that the same ExecutableModel can only safely support concurrent
/// calls to the 'execute' block if all state slots are read only.
/// TODO: GEX-549
class StateContext {
public:
  using CreateFn = void *(*)();
  using DestroyFn = void (*)(void *);

  /// Constructs a new state context with numSlots uninitialized state slots.
  explicit StateContext(size_t numSlots) : stateSlots(numSlots) {}

  /// Destroys all state slots in the state context. Slots are destroyed in
  /// reverse order since later slots may hold implicit references to earlier
  /// slots.
  ~StateContext() = default;

  /// Returns the number of state slots in the state context.
  size_t getNumStateSlots() const { return stateSlots.size(); }

  /// Returns true if the state slot at slotIndex has been set or emplaced.
  bool hasStateSlot(size_t slotIndex) {
    assert(slotIndex < stateSlots.size() &&
           "slotIndex must be within bounds when accessing state context");
    return stateSlots[slotIndex];
  }

  // No implicit copying is allowed.
  StateContext(const StateContext &rhs) = delete;
  StateContext &operator=(const StateContext &rhs) = delete;

  /// Moves an already constructed reference to T into the state slot
  /// at slotIndex. The state slot must not have been already set or emplaced.
  template <typename T>
  T &setStateSlot(size_t slotIndex, RCRef<T> payload) {
    assert(slotIndex < stateSlots.size() &&
           "slotIndex must be within bounds when accessing state context");
    assert(!stateSlots[slotIndex] &&
           "slotIndex has already been set or emplaced");
    stateSlots[slotIndex].reset<T>(std::move(payload));
    return stateSlots[slotIndex].get<T>();
  }

  /// Constructs a T in-place within the state slot at slotIndex. The state
  /// slot must not have already been set or emplaced.
  template <typename T, typename... Args>
  T &emplaceStateSlot(size_t slotIndex, Args &&...args) {
    assert(slotIndex < stateSlots.size() &&
           "slotIndex must be within bounds when accessing state context");
    assert(!stateSlots[slotIndex] &&
           "slotIndex has already been set or emplaced");
    stateSlots[slotIndex].reset<T>(
        RCRef<T>::create(std::forward<Args>(args)...));
    return stateSlots[slotIndex].get<T>();
  }

  /// Returns the state slot at slotIndex. The state slot must have already
  /// been emplaced.
  StateSlot &getStateSlot(size_t slotIndex) {
    assert(slotIndex < stateSlots.size() &&
           "slotIndex must be within bounds when accessing state context");
    assert(stateSlots[slotIndex] && "slotIndex has not been set or emplaced");
    return stateSlots[slotIndex];
  }

  /// Returns the payload in state slot at slotIndex. The state slot must
  /// have already been emplaced and be of type T.
  template <typename T>
  T &getStateSlotPayload(size_t slotIndex) {
    return getStateSlot(slotIndex).get<T>();
  }

  /// Returns the device graph cache, which is empty until Mojo creates it on
  /// first use (see `MGP_RT_GetOrCreateDeviceGraphCache`).
  ///
  /// Callers must hold `lockDeviceGraphCache()` while deciding whether to
  /// create the cache; this accessor does no locking of its own.
  MojoValue &getDeviceGraphCache(CreateFn create, DestroyFn destroy) {
    std::lock_guard<std::mutex> guard(deviceGraphCacheMutex);

    // A default-constructed MojoValue reports `isOwned()`, so a null
    // `getData()` is what distinguishes "not created yet".
    if (!deviceGraphCache.getData())
      deviceGraphCache = MojoValue::createOwningRef(create(), destroy);

    return deviceGraphCache;
  }

private:
  std::vector<StateSlot> stateSlots;

  /// Special slot designated for storing the device graph cache for the model.
  /// This cache is special as there is a single graph cache per-model, so is
  /// not indexed by a state slot.
  MojoValue deviceGraphCache;
  std::mutex deviceGraphCacheMutex;
};

} // namespace M::MLRT

#endif // MLRT_CORE_TENSOR_STATECONTEXT_H
