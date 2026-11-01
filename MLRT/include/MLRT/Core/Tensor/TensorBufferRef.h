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

#ifndef MLRT_CORE_TENSOR_TENSORBUFFERREF_H
#define MLRT_CORE_TENSOR_TENSORBUFFERREF_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "Support/Profiling/TimeProfiler.h"

#include "llvm/Support/Alignment.h"

#include <utility>

namespace M::MLRT {

// A block of owned memory. When there are no more references to this it will
// free the underlying memory.
struct OwnedMemoryBlock {
  OwnedMemoryBlock() = default;
  OwnedMemoryBlock(size_t size, std::optional<size_t> alignment,
                   AsyncRT::CompactCPUDevicePtr rt, void *takePtr = nullptr);
  ~OwnedMemoryBlock();
  void *data;
  size_t size;

  AsyncRT::CompactCPUDevicePtr cpuDevice;
};

/// Profiler entry for tracking the lifetimes of owned buffers.
using BufferRefLifetimeProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kMem, 3), Trace::kMem>;

/// A reference (aka smart pointer) to an underlying buffer of known size but
/// uninterpreted and possibly uninitialized bytes, along with metadata needed
/// to interop with the Modular cpuDevice. Each buffer has an async value to its
/// storage which will be freed on deconstruction.
class TensorBufferRef {
public:
  /// Constructs a reference in the 'null' state. Useful for references
  /// which are allocated lazily.
  TensorBufferRef() = default;

  /// Deallocates the underlying buffer if an owning reference.
  ~TensorBufferRef() = default;

  // Move is allowed for buffer references. The moved-from reference is
  // returned to the default 'null' state.
  TensorBufferRef(TensorBufferRef &&rhs) { swap(rhs); }
  TensorBufferRef &operator=(TensorBufferRef &&rhs) {
    swap(rhs);
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // Creation methods.
  //===--------------------------------------------------------------------===//

  /// Creates a TensorBufferRef with a given storage handle. The handle may be
  /// shared between other buffer refs will last as long as any use is alive.
  /// The storage handle must deallocate `buffer` on its destruction.
  static TensorBufferRef create(void *buffer, size_t size,
                                AsyncRT::AnyAsyncValueRef storageHandle,
                                std::optional<size_t> alignment);
  static TensorBufferRef create(void *buffer, size_t size,
                                AsyncRT::AnyAsyncValueRef storageHandle,
                                llvm::MaybeAlign alignment = std::nullopt);

  /// Create a buffer ref which does not track its memory and will not free the
  /// memory.
  static TensorBufferRef
  createWithNonTrackedMemory(void *buffer, size_t size,
                             std::optional<size_t> alignment = std::nullopt);

  /// Creates an 'owning' reference by allocating a new underlying buffer of
  /// size and alignment.
  static TensorBufferRef createOwned(AsyncRT::CompactCPUDevicePtr cpuDevice,
                                     size_t size,
                                     std::optional<size_t> alignment);
  /// Creates an 'owning' reference of size and alignment using takeFrom void*
  /// ptr that is allocated by caller. There is an assumption that the takeFrom
  /// pointer is allocated by caller via cpuDevice->getAllocator(). The
  /// allocated buffer will be freed by TensorBufferRef when the reference is
  /// destroyed. Since the ownership lies with TensorBufferRef, caller should
  /// not free anything that has been taken by TensorBufferRef.
  static TensorBufferRef take(AsyncRT::CompactCPUDevicePtr cpuDevice,
                              size_t size, void *takeFrom);
  //===--------------------------------------------------------------------===//
  // Core APIs for working with a TensorBufferRef.
  //===--------------------------------------------------------------------===//

  /// Returns a new TensorBufferRef pointing to the same data as this buffer.
  /// The new TensorBufferRef extends the storage lifetime of the memory.
  TensorBufferRef borrow() const;

  /// Returns a pointer to the underlying buffer. Returns nullptr if the
  /// reference is 'null' or has been moved out of. A zero sized buffer
  /// will not return nullptr.
  void *getBuffer() { return data; }
  const void *getBuffer() const { return data; }

  template <typename T>
  T *getBuffer() {
    return static_cast<T *>(getBuffer());
  }
  template <typename T>
  const T *getBuffer() const {
    return static_cast<const T *>(getBuffer());
  }

  /// Returns the size of the underlying buffer. Will return 0 if the
  /// reference is uninitialized or has been moved out of.
  size_t getSize() const { return size; }

  /// If the buffer is owning return the memory block which owns the memory.
  AsyncRT::AnyAsyncValueRef getMemStorageHandle() {
    if (!storageRef)
      return {};
    return storageRef.copy();
  }

  /// If the buffer is owning return the memory block which owns the memory.
  AsyncRT::AnyAsyncValueRef *getMemStorageHandleRef() {
    if (!storageRef)
      return nullptr;
    return &storageRef;
  }

  void printDebug(raw_ostream &os) const;
  void swap(TensorBufferRef &rhs);

private:
  /// Constructs a kOwned or kBorrowed reference.
  TensorBufferRef(void *buffer, size_t size,
                  AsyncRT::AnyAsyncValueRef storageRef,
                  llvm::MaybeAlign alignment,
                  BufferRefLifetimeProfilerEntry profilerEntry)
      : data(buffer), size(size), storageRef(std::move(storageRef)),
        profilerEntry(std::move(profilerEntry)) {}

  // Implicit copying is not allowed.
  TensorBufferRef(const TensorBufferRef &rhs) = delete;
  TensorBufferRef &operator=(const TensorBufferRef &rhs) = delete;

  // Reset the reference to the kNull state. However only the ownership,
  // buffer/object, and size fields are reset.
  void reset();

  /// The data contained in the buffer
  void *data = nullptr;

  /// The size of the underlying buffer.
  size_t size = 0;

  /// Tracks the storage object which preserves the lifetime of data.
  AsyncRT::AnyAsyncValueRef storageRef;

  BufferRefLifetimeProfilerEntry profilerEntry;
};

inline raw_ostream &operator<<(raw_ostream &os, const TensorBufferRef &value) {
  value.printDebug(os);
  return os;
}

} // namespace M::MLRT

namespace std {

// For ADL style swap.
template <>
inline void swap(M::MLRT::TensorBufferRef &lhs, M::MLRT::TensorBufferRef &rhs) {
  lhs.swap(rhs);
}

} // namespace std

#endif // MLRT_CORE_TENSOR_TENSORBUFFERREF_H
