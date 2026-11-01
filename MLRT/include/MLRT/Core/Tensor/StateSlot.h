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

#ifndef MLRT_CORE_TENSOR_STATESLOT_H
#define MLRT_CORE_TENSOR_STATESLOT_H

#include "AsyncRT/ForwardDecls.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "Support/ADT/GenericRCRef.h"
#include "Support/ErrorOr.h"
#include "Support/ReferenceCounted.h"
#include "Support/TypeID.h"

namespace M::MLRT {

/// A type-erased wrapper for all context 'state slots'. A state slot holds an
/// arbitrary reference counted payload (which should derive from
/// ReferenceCounted). See TensorStateSlot for a payload that caches a buffer
/// allocation.
class StateSlot {
public:
  /// Constructs the empty state slot.
  StateSlot() = default;

  /// Establishes state slot to take reference to payload.
  template <typename T>
  void reset(RCRef<T> &&newPayload) {
    payload = GenericRCRef::fromRCRef<T>(std::move(newPayload));
  }

  /// Clears the state slot and returns it to its original empty state
  void clear() {
    // Copy-and-swap idiom to replace with an empty GenericRCRef
    GenericRCRef empty;
    std::swap(payload, empty);
  }

  /// Destroy this state slot, dropping the reference to its payload.
  ~StateSlot() = default;

  /// Returns true if have a payload.
  operator bool() const { return payload; }

  /// Returns the payload of type T of this state slot. The static type id for T
  /// must match the type id of the state slot.
  template <typename T>
  T &get() {
    assert(payload && "StateSlot has no payload");
    return *payload.get<T>();
  }

  /// Returns a buffer reference to a buffer of size with alignment.
  /// If a buffer with matching size already exists at bufferIndex in this state
  /// slot then it will be returned. Otherwise, a fresh buffer is allocated
  /// and returned, and any previously cached buffer is freed.
  ///
  /// The result will always be a 'borrow' of the underlying cached buffer.
  /// This means:
  ///  - uses of the returned reference must not outlive the overall context
  ///    holding this state slot.
  ///  - no other call to this method for the same state slot and buffer index
  ///    should be made while the returned reference is still in use.
  TensorBufferRef getOrCreateBuffer(size_t bufferIndex,
                                    AsyncRT::CompactCPUDevicePtr cpuDevice,
                                    size_t size,
                                    std::optional<size_t> alignment);

  /// Returns a reference to the already allocated and cached buffer at
  /// bufferIndex in this state slot. Returns Error if no buffer is cached at
  /// given index.
  ///
  /// The result will always be a 'borrow' of the underlying cached buffer.
  /// See getOrCreateBuffer for safety conditions.
  ErrorOr<TensorBufferRef> getBuffer(size_t bufferIndex);
  ErrorOr<TensorBufferRef &> getBufferRef(size_t bufferIndex);

  /// Sets the cached buffer at bufferIndex to the given bufferRef. Returns
  /// error if bufferIndex already contains a buffer.
  ///
  /// If bufferRef is owned we simply take over that ownership. However if
  /// it is borrowed then it must have lifetime of at least 'execute'
  /// so that similarly 'execute'-scoped borrows may be taken from it.
  ErrorOrSuccess setBuffer(size_t bufferIndex, TensorBufferRef bufferRef);

  /// Returns the type id for the value held by this state slot.
  TypeID getTypeID() const { return payload.getTypeID(); }

  /// Returns the underlying payload pointer held by this state slot.
  void *getUnderlyingPointer() { return payload.getOpaquePointer(); }

private:
  /// The state slot payload.
  GenericRCRef payload;
};

/// A state slot payload that caches a single buffer allocation. Tensors and
/// other more refined types can be constructed from this buffer. Used by
/// `mgp.buffer.cache` / `mgp.buffer.get_cached` to cache buffer allocations
/// across calls to the 'execute' block via the !mgp.context cache.
class TensorStateSlot : public ReferenceCounted<TensorStateSlot> {
public:
  /// Returns a buffer reference to a buffer of size with alignment.
  /// If a buffer with matching size already exists in this state slot then it
  /// will be returned. Otherwise, a fresh buffer is allocated and returned, and
  /// any previously cached buffer is freed.
  TensorBufferRef getOrCreateBuffer(AsyncRT::CompactCPUDevicePtr cpuDevice,
                                    size_t size,
                                    std::optional<size_t> alignment);

  /// Returns a reference to the already allocated and cached buffer in this
  /// state slot. Returns Error if no buffer is cached.
  ErrorOr<TensorBufferRef> getBuffer();
  ErrorOr<TensorBufferRef &> getBufferRef();

  /// Sets the cached buffer to the given bufferRef. Returns error if a buffer
  /// has already been cached.
  ErrorOrSuccess setBuffer(TensorBufferRef bufferRef);

private:
  /// Ensure the buffer in entry is allocated with size, reallocating
  /// any existing buffer if required.
  static void updateCacheIfNeeded(TensorBufferRef &entry,
                                  AsyncRT::CompactCPUDevicePtr cpuDevice,
                                  size_t size, std::optional<size_t> alignment);

  /// A place to hold a pre-allocated or cached buffer. A default constructed
  /// (null) reference indicates no buffer has been cached.
  TensorBufferRef bufferCache;
};

/// A state slot payload that holds a single size_t (index) value.
/// Used by `mgp.int.cache` / `mgp.int.get_cached` to convey index
/// values (e.g. dynamic dimension values) from the 'init' block to the
/// 'execute' block via the !mgp.context cache.
class IntStateSlot : public ReferenceCounted<IntStateSlot> {
public:
  explicit IntStateSlot(size_t value) : value(value) {}

  size_t getValue() const { return value; }

private:
  size_t value;
};

} // namespace M::MLRT

#endif // MLRT_CORE_TENSOR_STATESLOT_H
