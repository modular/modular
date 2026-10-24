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

#ifndef SDK_DRIVERAPI_PYTHON_BUFFERSTORAGE_H
#define SDK_DRIVERAPI_PYTHON_BUFFERSTORAGE_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace M::Driver::Python {

/// Allocation intent for a Buffer. `max.driver` owns this flag set and its
/// per-backend mapping; a flag must change how memory is allocated.
enum class Usage : uint32_t {
  Default = 0,
  /// Host memory for staging transfers, page-locked on some backends.
  Staging = 1u << 0,
  // Host memory needs no tracking, only useful when paired with Staging.
  Untracked = 1u << 1,
};

inline constexpr Usage operator|(Usage a, Usage b) {
  return static_cast<Usage>(static_cast<uint32_t>(a) |
                            static_cast<uint32_t>(b));
}

inline constexpr bool usageContains(Usage value, Usage flag) {
  return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

/// Backing store for a `Buffer`: a refcounted MLRT `DeviceBuffer` plus the
/// `TensorSpec` describing the region of it this storage represents.
///
/// Replaces the GenericML `DeviceMemory` wrapper Buffer used to hold
/// (DRIV-399/MAX-776). Multiple `Buffer`s can share one `BufferStorage`
/// instance (e.g. slices produced by `Buffer::get`), which is why this is
/// always held behind a `shared_ptr`.
struct BufferStorage {
  BufferStorage(M::Driver::DeviceBufferRef buf, TensorSpec spec,
                Usage usage = Usage::Default);

  void *data() const;
  size_t bytecount() const;
  const TensorSpec &getSpec() const;

  /// A device is host-resident either because its buffer is itself pinned
  /// host memory, or because the owning context's device is the CPU.
  bool isHost() const;

  bool isPinned() const;

  ErrorOrSuccess memset(int value);

  M::Driver::DeviceBufferRef buf;
  TensorSpec spec;

  /// Allocation-intent recorded at construction (MXF-635 R1). Shared by every
  /// `Buffer` over this storage, including slices and views.
  Usage usage = Usage::Default;
};

/// Mirrors GenericML `DeviceMemory::createView`: bounds-checks `newSpec`
/// against `storage`'s own spec and either shares the existing buffer
/// (when the view is untouched) or creates a byte-range sub-buffer.
ErrorOr<std::shared_ptr<BufferStorage>>
createStorageView(const BufferStorage &storage, const TensorSpec &newSpec,
                  size_t byteOffset = 0);

/// Views the whole storage on `targetCtx`, ordered after `after`.
ErrorOr<std::shared_ptr<BufferStorage>>
createStorageViewOn(const BufferStorage &storage,
                    const M::Driver::DeviceContextRef &targetCtx,
                    M::Driver::DeviceEvent &after);

/// Wraps the same memory in a non-owning `DeviceBuffer`, so the result aliases
/// `storage` without extending its lifetime.
std::shared_ptr<BufferStorage>
createBorrowedStorage(const std::shared_ptr<BufferStorage> &storage);

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_BUFFERSTORAGE_H
