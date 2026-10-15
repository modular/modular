//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines a slab-backed bump allocator. It is a modernized version of
// llvm::BumpPtrAllocator. It would be great to eventually upstream and unify.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BUMPALLOCATOR_H
#define SUPPORT_BUMPALLOCATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AllocatorBase.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace M {

/// Allocates memory in a monotonically growing, slab-backed pool.
///
/// Individual allocations cannot be freed. All allocations remain valid until
/// the allocator is reset or destroyed. Allocations larger than the size
/// threshold receive dedicated slabs.
class BumpAllocator {
public:
  static constexpr size_t slabSize = 4096;
  static constexpr size_t sizeThreshold = slabSize;
  static constexpr size_t growthDelay = 128;
  static constexpr size_t minAlignment = 8;

  static_assert(sizeThreshold <= slabSize);
  static_assert(growthDelay > 0);
  static_assert(minAlignment > 0 && (minAlignment & (minAlignment - 1)) == 0);
  static_assert(minAlignment <= alignof(std::max_align_t));

  BumpAllocator() = default;
  BumpAllocator(const BumpAllocator &) = delete;
  BumpAllocator &operator=(const BumpAllocator &) = delete;
  BumpAllocator(BumpAllocator &&other) noexcept;
  BumpAllocator &operator=(BumpAllocator &&other) noexcept;
  ~BumpAllocator();

  /// Rolls back allocations when it leaves scope.
  ///
  /// Checkpoints must be destroyed in LIFO order and must not outlive or move
  /// their allocator.
  class [[nodiscard]] Checkpoint {
  public:
    Checkpoint(const Checkpoint &) = delete;
    Checkpoint &operator=(const Checkpoint &) = delete;
    Checkpoint(Checkpoint &&) = delete;
    Checkpoint &operator=(Checkpoint &&) = delete;
    ~Checkpoint();

  private:
    friend class BumpAllocator;
    explicit Checkpoint(BumpAllocator &allocator);

    BumpAllocator *allocator;
    size_t slabIndex;
    char *pointer;
    size_t activeCustomSlabCount;
    size_t depth;
  };

  /// Creates a scoped rollback point for subsequent allocations.
  Checkpoint checkpoint() { return Checkpoint(*this); }

  /// Frees all dedicated slabs and all but the first regular slab, then resets
  /// the allocation pointer to the beginning of the retained slab.
  void reset();

  /// Allocates `size` bytes with the requested alignment.
  ///
  /// A zero-sized allocation is valid and returns a non-null pointer that must
  /// not be dereferenced.
  LLVM_ATTRIBUTE_RETURNS_NONNULL void *allocate(size_t size,
                                                llvm::Align alignment);

  /// Allocates `size` bytes with the requested power-of-two alignment.
  LLVM_ATTRIBUTE_RETURNS_NONNULL void *allocate(size_t size, size_t alignment) {
    assert(alignment > 0 && "alignment must be nonzero");
    return allocate(size, llvm::Align(alignment));
  }

  /// Allocates uninitialized storage for `count` objects of type `T`.
  template <typename T>
  LLVM_ATTRIBUTE_RETURNS_NONNULL T *allocate(size_t count = 1) {
    return static_cast<T *>(allocate(count * sizeof(T), alignof(T)));
  }

  /// Allocates and constructs an object whose destructor will not be invoked.
  template <typename T, typename... Args>
  T *create(Args &&...args) {
    return new (allocate<T>()) T(std::forward<Args>(args)...);
  }

  /// Copy-constructs a value in allocator-owned storage.
  template <typename T>
  T *intern(const T &value) {
    static_assert(std::is_trivially_destructible_v<T>);
    return new (allocate<T>()) T(value);
  }

  /// Move-constructs a value in allocator-owned storage.
  template <typename T>
  std::enable_if_t<!std::is_lvalue_reference_v<T>, std::remove_cv_t<T> *>
  intern(T &&value) {
    using ValueType = std::remove_cv_t<T>;
    static_assert(std::is_trivially_destructible_v<ValueType>);
    return new (allocate<ValueType>()) ValueType(std::forward<T>(value));
  }

  /// Copy-constructs an array in allocator-owned storage.
  template <typename T>
  llvm::ArrayRef<T> internElements(llvm::ArrayRef<T> values) {
    static_assert(std::is_trivially_destructible_v<T>);
    if (values.empty())
      return {};
    assert(values.size() <= std::numeric_limits<size_t>::max() / sizeof(T) &&
           "array allocation size overflow");
    T *storage = allocate<T>(values.size());
    std::uninitialized_copy(values.begin(), values.end(), storage);
    return {storage, values.size()};
  }

  /// Copies a string into allocator-owned storage.
  llvm::StringRef internString(llvm::StringRef value) {
    llvm::ArrayRef<char> storage =
        internElements<char>({value.data(), value.size()});
    return {storage.data(), storage.size()};
  }

  /// Marks an allocation inaccessible to AddressSanitizer. Storage is not
  /// reclaimed until reset or destruction.
  void deallocate(const void *ptr, size_t size, size_t alignment);

  /// Marks storage for `count` objects inaccessible to AddressSanitizer.
  template <typename T>
  void deallocate(T *ptr, size_t count = 1) {
    deallocate(ptr, count * sizeof(T), alignof(T));
  }

  /// Returns the number of regular and dedicated slabs.
  size_t getNumSlabs() const { return slabs.size() + customSizedSlabs.size(); }

  /// Returns a stable identifier for `ptr`, or std::nullopt if it is not owned.
  ///
  /// Identifiers for objects in dedicated slabs are negative.
  std::optional<int64_t> identifyObject(const void *ptr) const;

  /// Returns a stable identifier for an allocator-owned pointer.
  int64_t identifyKnownObject(const void *ptr) const;

  /// Returns a stable identifier scaled by `T`'s alignment.
  template <typename T>
  int64_t identifyKnownAlignedObject(const void *ptr) const {
    int64_t result = identifyKnownObject(ptr);
    assert(result % alignof(T) == 0 && "incorrect alignment information");
    return result / alignof(T);
  }

  /// Returns the total number of bytes reserved by all slabs.
  size_t getTotalMemory() const;

  /// Sets the sanitizer red-zone size when ABI checks are enabled.
  void setRedZoneSize([[maybe_unused]] size_t size) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    redZoneSize = size;
#endif
  }

  /// Prints allocator statistics to stderr.
  void printStats() const;

private:
  static constexpr size_t noSlab = std::numeric_limits<size_t>::max();

  struct CustomSizedSlab {
    void *pointer;
    size_t size;
  };

  LLVM_ATTRIBUTE_RETURNS_NONNULL LLVM_ATTRIBUTE_NOINLINE void *
  allocateSlow(size_t size, size_t sizeToAllocate, llvm::Align alignment);

  static size_t computeSlabSize(size_t slabIndex);
  void startNewSlab();
  void deallocateSlabs(size_t beginIndex);
  void deallocateCustomSizedSlabs();
  void clear();
  void rollback(Checkpoint &checkpoint);

  llvm::MallocAllocator backingAllocator;

  /// Points to the next free byte in the current slab.
  char *currentPointer = nullptr;

  size_t currentSlabIndex = noSlab;

  /// One past the current slab's end, plus one. Zero indicates no current slab.
  uintptr_t endSentinel = 0;

  llvm::SmallVector<void *, 4> slabs;
  llvm::SmallVector<CustomSizedSlab, 0> customSizedSlabs;
  llvm::SmallVector<size_t, 0> activeCustomSlabs;
  llvm::SmallVector<size_t, 0> availableCustomSlabs;
  size_t checkpointDepth = 0;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  size_t redZoneSize = 1;
#endif
};

} // namespace M

#endif // SUPPORT_BUMPALLOCATOR_H
