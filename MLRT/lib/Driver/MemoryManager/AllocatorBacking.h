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
// This file declares the interfaces through which a `MemoryManager` obtains
// physical backing and memory info from a device. The manager may hold its
// mutex, `mu_`, across any of these calls, so an implementation must not call
// back into the manager.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_MEMORYMANAGER_ALLOCATORBACKING_H
#define MLRT_LIB_DRIVER_MEMORYMANAGER_ALLOCATORBACKING_H

#include "Support/ErrorOr.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

namespace M::Driver {

// Total and free bytes, of a device or of a manager's budget.
struct AllocatorMemInfo {
  size_t total = 0;
  // Unset when the device cannot measure free memory.
  std::optional<size_t> free;
};

// Plain chunk backing and the device's memory info.
//
// An implementation's destructor must not call into the driver: a manager can
// outlive the backend of the device it serves.
class AllocatorChunks {
public:
  virtual ~AllocatorChunks() = default;

  // REQUIRES: byteSize > 0
  virtual ErrorOrSuccess chunkAlloc(void **ptr, size_t byteSize,
                                    bool isHost) = 0;
  virtual ErrorOrSuccess chunkFree(void *ptr, bool isHost) = 0;

  virtual ErrorOr<AllocatorMemInfo> getMemoryInfo() const = 0;
};

// Virtual memory backing. All sizes are multiples of the device's VMM
// granularity.
class AllocatorVmm {
public:
  virtual ~AllocatorVmm() = default;

  // Address space, with no physical backing.
  virtual ErrorOr<void *> vmmAddressReserve(size_t byteSize) = 0;
  virtual ErrorOrSuccess vmmAddressFree(void *va, size_t byteSize) = 0;

  // One physical granule. The handle is opaque, and 0 is invalid.
  virtual ErrorOr<uint64_t> vmmCreate(size_t byteSize) = 0;
  virtual ErrorOrSuccess vmmRelease(uint64_t handle) = 0;

  // Mapping. `vmmMap` may leave access to a later `vmmSetAccess`, so a caller
  // mapping many granules grants access once. A backend that does not split
  // the two grants access in `vmmMap` and makes `vmmSetAccess` a no-op. A
  // failed `vmmMapAndSetAccess` leaves nothing mapped; a failed `vmmSetAccess`
  // leaves the range mapped.
  virtual ErrorOrSuccess vmmMapAndSetAccess(void *va, size_t byteSize,
                                            uint64_t handle) = 0;
  virtual ErrorOrSuccess vmmUnmap(void *va, size_t byteSize) = 0;
  virtual ErrorOrSuccess vmmMap(void *va, size_t byteSize, uint64_t handle) = 0;
  virtual ErrorOrSuccess vmmSetAccess(void *va, size_t byteSize) = 0;
};

// A manager's backing. A manager is plain or VMM for its whole life, so it
// holds exactly one of the two.
using AllocatorBacking = std::variant<std::unique_ptr<AllocatorChunks>,
                                      std::unique_ptr<AllocatorVmm>>;

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_MEMORYMANAGER_ALLOCATORBACKING_H
