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
//
// This file declares the interface through which a `MemoryManagerFreeList`
// reaches the context it serves.
//
// A `MemoryManager` guards its own state, and the state of every free list
// that draws from it, with one mutex: `MemoryManager::mu_`. Allocation, free,
// and reclaim all take it, from any thread that uses the manager. Each method
// below says whether its caller holds `mu_`. A method whose caller may hold
// `mu_` must not call back into the manager, because taking `mu_` again on the
// same thread deadlocks.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_MEMORYMANAGER_ALLOCATORCONTEXT_H
#define MLRT_LIB_DRIVER_MEMORYMANAGER_ALLOCATORCONTEXT_H

#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

#include <cstddef>
#include <cstdint>

namespace M::Driver {

// A point in a context's stream. The manager holds these in its pending frees
// and retiring ranges, which can outlive the context that recorded them. The
// manager queries them with `mu_` released.
class AllocatorEvent : public ReferenceCounted<AllocatorEvent> {
public:
  virtual ~AllocatorEvent() = default;

  virtual ErrorOr<bool> isReady() const = 0;
  virtual ErrorOrSuccess synchronize() = 0;
};

using AllocatorEventRef = RCRef<AllocatorEvent>;

// The context a free list serves: its stream, and the driver calls the free
// list and its manager make on the stream's behalf. A free list with no
// context is the root list of its manager.
//
// An implementation's destructor must not call into the driver: free lists die
// inside `~DeviceContext`, after the backend subclass is gone.
class AllocatorContext {
public:
  // Each enumerator has the value of its counterpart in `Device::AllocMode` or
  // `Device::FreeKind`, so the legacy driver converts with a `static_cast`.
  // Give a new enumerator its counterpart's value, and add a `static_assert`
  // for the pair in DeviceAllocator.cpp.

  // How a fallthrough allocation is obtained.
  enum class AllocMode : uint8_t {
    HOST = 1,
    SYNC = 2,
    ASYNC = 3,
  };

  // How a fallthrough allocation is released.
  enum class FreeKind : uint8_t {
    HOST = 1,
    DEVICE = 2,
  };

  virtual ~AllocatorContext() = default;

  // Stream ordering.

  // Caller holds `mu_`.
  virtual ErrorOr<AllocatorEventRef> enqueueEvent() = 0;
  // Returns a null ref when the stream is its own base. Caller holds `mu_`.
  virtual ErrorOr<AllocatorEventRef> enqueueBaseStreamEvent() = 0;
  // Caller holds `mu_`. In debug sync mode, the CUDA context waits for its
  // stream here; the wait does not call back, so it cannot deadlock.
  // REQUIRES: this context recorded `event`.
  virtual ErrorOrSuccess enqueueWaitFor(AllocatorEvent &event) = 0;

  // Capture state. Allocation and reclaim take different paths while a graph
  // is being captured on the stream or on its base stream.

  // Caller may hold `mu_`.
  virtual bool isGraphCapturing() const = 0;
  // Caller may hold `mu_`.
  virtual bool isBaseStreamCapturing() const = 0;

  // Reclaim hooks, called on an allocation miss to return memory that is still
  // held by completed work.

  // Caller must NOT hold `mu_`: the event handlers this runs take it.
  virtual ErrorOrSuccess pollPendingEvents() = 0;
  // Returns whether any idle stream view was flushed. Caller must NOT hold
  // `mu_`: a flush frees into the manager.
  virtual bool flushIdleStreamViews() = 0;

  // Driver fallthrough, for requests the memory manager cannot serve.

  // Caller must NOT hold `mu_`.
  virtual ErrorOrSuccess memAlloc(void **ptr, size_t byteSize,
                                  AllocMode mode) = 0;
  // Caller must NOT hold `mu_`.
  virtual ErrorOrSuccess memFree(void *ptr, FreeKind kind) = 0;

  // Debug memory ops, used to write and check redzone and poison patterns.

  // Caller must NOT hold `mu_`.
  virtual ErrorOrSuccess enqueueSetMemory(void *dst, size_t byteSize,
                                          uint64_t val, size_t valSize) = 0;
  // Caller must NOT hold `mu_`.
  virtual ErrorOrSuccess enqueueCopyFromDevice(void *dst, const void *src,
                                               size_t byteSize) = 0;
  // Caller must NOT hold `mu_`: this polls pending events, whose handlers take
  // it.
  virtual ErrorOrSuccess synchronize() = 0;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_MEMORYMANAGER_ALLOCATORCONTEXT_H
