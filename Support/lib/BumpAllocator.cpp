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

#include "Support/BumpAllocator.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <utility>

using namespace M;

BumpAllocator::BumpAllocator(BumpAllocator &&other) noexcept
    : backingAllocator(std::move(other.backingAllocator)),
      currentPointer(other.currentPointer),
      currentSlabIndex(other.currentSlabIndex), endSentinel(other.endSentinel),
      slabs(std::move(other.slabs)),
      customSizedSlabs(std::move(other.customSizedSlabs)),
      activeCustomSlabs(std::move(other.activeCustomSlabs)),
      availableCustomSlabs(std::move(other.availableCustomSlabs)) {
  assert(other.checkpointDepth == 0 &&
         "cannot move an allocator with active checkpoints");
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  redZoneSize = other.redZoneSize;
#endif
  other.currentPointer = nullptr;
  other.currentSlabIndex = noSlab;
  other.endSentinel = 0;
  other.slabs.clear();
  other.customSizedSlabs.clear();
  other.activeCustomSlabs.clear();
  other.availableCustomSlabs.clear();
}

BumpAllocator &BumpAllocator::operator=(BumpAllocator &&other) noexcept {
  assert(checkpointDepth == 0 &&
         "cannot replace an allocator with active checkpoints");
  assert(other.checkpointDepth == 0 &&
         "cannot move an allocator with active checkpoints");
  if (this == &other)
    return *this;
  clear();

  backingAllocator = std::move(other.backingAllocator);
  currentPointer = other.currentPointer;
  currentSlabIndex = other.currentSlabIndex;
  endSentinel = other.endSentinel;
  slabs = std::move(other.slabs);
  customSizedSlabs = std::move(other.customSizedSlabs);
  activeCustomSlabs = std::move(other.activeCustomSlabs);
  availableCustomSlabs = std::move(other.availableCustomSlabs);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  redZoneSize = other.redZoneSize;
#endif

  other.currentPointer = nullptr;
  other.currentSlabIndex = noSlab;
  other.endSentinel = 0;
  other.slabs.clear();
  other.customSizedSlabs.clear();
  other.activeCustomSlabs.clear();
  other.availableCustomSlabs.clear();
  return *this;
}

BumpAllocator::~BumpAllocator() {
  assert(checkpointDepth == 0 && "allocator destroyed before its checkpoints");
  clear();
}

BumpAllocator::Checkpoint::Checkpoint(BumpAllocator &allocator)
    : allocator(&allocator), slabIndex(allocator.currentSlabIndex),
      pointer(allocator.currentPointer),
      activeCustomSlabCount(allocator.activeCustomSlabs.size()),
      depth(++allocator.checkpointDepth) {}

BumpAllocator::Checkpoint::~Checkpoint() { allocator->rollback(*this); }

/// Frees all dedicated slabs and all but the first regular slab, then resets
/// the allocation pointer to the beginning of the retained slab.
void BumpAllocator::reset() {
  assert(checkpointDepth == 0 && "cannot reset with active checkpoints");
  deallocateCustomSizedSlabs();
  customSizedSlabs.clear();
  activeCustomSlabs.clear();
  availableCustomSlabs.clear();

  if (slabs.empty()) {
    currentPointer = nullptr;
    currentSlabIndex = noSlab;
    endSentinel = 0;
    return;
  }

  currentPointer = static_cast<char *>(slabs.front());
  currentSlabIndex = 0;
  endSentinel = reinterpret_cast<uintptr_t>(currentPointer) + slabSize + 1;

  __asan_poison_memory_region(slabs.front(), computeSlabSize(0));
  deallocateSlabs(/*beginIndex=*/1);
  slabs.resize(1);
}

/// Allocates `size` bytes with the requested alignment.
///
/// A zero-sized allocation is valid and returns a non-null pointer that must
/// not be dereferenced.
void *BumpAllocator::allocate(size_t size, llvm::Align alignment) {
  size_t sizeToAllocate = size;
#if LLVM_ADDRESS_SANITIZER_BUILD && LLVM_ENABLE_ABI_BREAKING_CHECKS
  sizeToAllocate += redZoneSize;
#endif
  sizeToAllocate =
      llvm::alignToPowerOf2(sizeToAllocate, BumpAllocator::minAlignment);

  uintptr_t alignedPointer = reinterpret_cast<uintptr_t>(currentPointer);
  if (alignment.value() > minAlignment)
    alignedPointer = llvm::alignAddr(currentPointer, alignment);
  uintptr_t allocationEnd = alignedPointer + sizeToAllocate;
  assert(allocationEnd >= reinterpret_cast<uintptr_t>(currentPointer) &&
         "alignment plus size must not overflow");

  if (LLVM_LIKELY(allocationEnd < endSentinel)) {
    currentPointer = reinterpret_cast<char *>(allocationEnd);
    __msan_allocated_memory(reinterpret_cast<char *>(alignedPointer), size);
    __asan_unpoison_memory_region(reinterpret_cast<char *>(alignedPointer),
                                  size);
    return reinterpret_cast<char *>(alignedPointer);
  }

  return allocateSlow(size, sizeToAllocate, alignment);
}

void *BumpAllocator::allocateSlow(size_t size, size_t sizeToAllocate,
                                  llvm::Align alignment) {
  size_t paddedSize = sizeToAllocate + alignment.value() - 1;
  if (paddedSize > sizeThreshold) {
    size_t slabIndex = noSlab;
    for (size_t position = 0; position < availableCustomSlabs.size();
         ++position) {
      size_t candidate = availableCustomSlabs[position];
      if (customSizedSlabs[candidate].size < paddedSize)
        continue;
      slabIndex = candidate;
      availableCustomSlabs.erase(availableCustomSlabs.begin() + position);
      break;
    }

    if (slabIndex == noSlab) {
      void *newSlab =
          backingAllocator.Allocate(paddedSize, alignof(std::max_align_t));
      __asan_poison_memory_region(newSlab, paddedSize);
      slabIndex = customSizedSlabs.size();
      customSizedSlabs.push_back({newSlab, paddedSize});
    }
    activeCustomSlabs.push_back(slabIndex);

    CustomSizedSlab &slab = customSizedSlabs[slabIndex];
    uintptr_t alignedAddress = llvm::alignAddr(slab.pointer, alignment);
    assert(alignedAddress + size <=
           reinterpret_cast<uintptr_t>(slab.pointer) + slab.size);
    char *alignedPointer = reinterpret_cast<char *>(alignedAddress);
    __msan_allocated_memory(alignedPointer, size);
    __asan_unpoison_memory_region(alignedPointer, size);
    return alignedPointer;
  }

  startNewSlab();
  uintptr_t alignedAddress = llvm::alignAddr(currentPointer, alignment);
  assert(alignedAddress + sizeToAllocate < endSentinel &&
         "unable to satisfy allocation");
  char *alignedPointer = reinterpret_cast<char *>(alignedAddress);
  currentPointer = alignedPointer + sizeToAllocate;
  __msan_allocated_memory(alignedPointer, size);
  __asan_unpoison_memory_region(alignedPointer, size);
  return alignedPointer;
}

/// Marks an allocation inaccessible to AddressSanitizer. Storage is not
/// reclaimed until reset or destruction.
void BumpAllocator::deallocate(const void *ptr, size_t size,
                               size_t /*alignment*/) {
  __asan_poison_memory_region(ptr, size);
}

/// Returns a stable identifier for `ptr`, or std::nullopt if it is not owned.
///
/// Identifiers for objects in dedicated slabs are negative.
std::optional<int64_t>
BumpAllocator::identifyObject(const void *pointer) const {
  const char *target = static_cast<const char *>(pointer);
  int64_t slabOffset = 0;
  for (size_t index = 0; index < slabs.size(); ++index) {
    const char *slab = static_cast<const char *>(slabs[index]);
    size_t size = computeSlabSize(index);
    if (target >= slab && target < slab + size)
      return slabOffset + static_cast<int64_t>(target - slab);
    slabOffset += static_cast<int64_t>(size);
  }

  int64_t customSlabOffset = -1;
  for (const auto &[pointer, size] : customSizedSlabs) {
    const char *slab = static_cast<const char *>(pointer);
    if (target >= slab && target < slab + size)
      return customSlabOffset - static_cast<int64_t>(target - slab);
    customSlabOffset -= static_cast<int64_t>(size);
  }
  return std::nullopt;
}

/// Returns a stable identifier for an allocator-owned pointer.
int64_t BumpAllocator::identifyKnownObject(const void *pointer) const {
  std::optional<int64_t> result = identifyObject(pointer);
  assert(result && "pointer is not owned by this allocator");
  return *result;
}

/// Returns the total number of bytes reserved by all slabs.
size_t BumpAllocator::getTotalMemory() const {
  size_t totalMemory = 0;
  for (size_t index = 0; index < slabs.size(); ++index)
    totalMemory += computeSlabSize(index);
  for (const auto &[pointer, size] : customSizedSlabs) {
    static_cast<void>(pointer);
    totalMemory += size;
  }
  return totalMemory;
}

/// Prints allocator statistics to stderr.
void BumpAllocator::printStats() const {
  llvm::errs() << "\nNumber of memory regions: " << getNumSlabs() << '\n'
               << "Bytes allocated: " << getTotalMemory() << '\n'
               << " (includes alignment, etc)\n";
}

size_t BumpAllocator::computeSlabSize(size_t slabIndex) {
  return slabSize *
         (size_t{1} << std::min<size_t>(30, slabIndex / growthDelay));
}

void BumpAllocator::startNewSlab() {
  size_t nextSlabIndex = currentSlabIndex == noSlab ? 0 : currentSlabIndex + 1;
  if (nextSlabIndex == slabs.size()) {
    size_t allocatedSlabSize = computeSlabSize(nextSlabIndex);
    void *newSlab =
        backingAllocator.Allocate(allocatedSlabSize, alignof(std::max_align_t));
    __asan_poison_memory_region(newSlab, allocatedSlabSize);
    slabs.push_back(newSlab);
  }

  assert(nextSlabIndex < slabs.size());
  currentSlabIndex = nextSlabIndex;
  currentPointer = static_cast<char *>(slabs[currentSlabIndex]);
  endSentinel = reinterpret_cast<uintptr_t>(currentPointer) +
                computeSlabSize(currentSlabIndex) + 1;
}

void BumpAllocator::deallocateSlabs(size_t beginIndex) {
  for (size_t index = beginIndex; index < slabs.size(); ++index) {
    backingAllocator.Deallocate(slabs[index], computeSlabSize(index),
                                alignof(std::max_align_t));
  }
}

void BumpAllocator::deallocateCustomSizedSlabs() {
  for (const auto &[pointer, size] : customSizedSlabs)
    backingAllocator.Deallocate(pointer, size, alignof(std::max_align_t));
}

void BumpAllocator::rollback(Checkpoint &checkpoint) {
  assert(checkpoint.depth == checkpointDepth &&
         "checkpoints must be rolled back in LIFO order");
  assert(checkpoint.activeCustomSlabCount <= activeCustomSlabs.size());

  for (size_t position = checkpoint.activeCustomSlabCount;
       position < activeCustomSlabs.size(); ++position) {
    size_t slabIndex = activeCustomSlabs[position];
    [[maybe_unused]] const CustomSizedSlab &slab = customSizedSlabs[slabIndex];
    __asan_poison_memory_region(slab.pointer, slab.size);
    availableCustomSlabs.push_back(slabIndex);
  }
  activeCustomSlabs.resize(checkpoint.activeCustomSlabCount);

  if (checkpoint.slabIndex == noSlab) {
    if (currentSlabIndex != noSlab) {
      for (size_t index = 0; index <= currentSlabIndex; ++index) {
        char *slabBegin = static_cast<char *>(slabs[index]);
        [[maybe_unused]] size_t usedSize = index == currentSlabIndex
                                               ? currentPointer - slabBegin
                                               : computeSlabSize(index);
        __asan_poison_memory_region(slabBegin, usedSize);
      }
    }
    currentPointer = nullptr;
    currentSlabIndex = noSlab;
    endSentinel = 0;
  } else {
    assert(checkpoint.slabIndex < slabs.size());
    assert(currentSlabIndex >= checkpoint.slabIndex);
    char *slabBegin = static_cast<char *>(slabs[checkpoint.slabIndex]);
    char *slabEnd = slabBegin + computeSlabSize(checkpoint.slabIndex);
    assert(checkpoint.pointer >= slabBegin && checkpoint.pointer <= slabEnd);
    if (currentSlabIndex == checkpoint.slabIndex) {
      assert(currentPointer >= checkpoint.pointer);
      __asan_poison_memory_region(checkpoint.pointer,
                                  currentPointer - checkpoint.pointer);
    } else {
      __asan_poison_memory_region(checkpoint.pointer,
                                  slabEnd - checkpoint.pointer);
      for (size_t index = checkpoint.slabIndex + 1; index <= currentSlabIndex;
           ++index) {
        char *laterSlabBegin = static_cast<char *>(slabs[index]);
        [[maybe_unused]] size_t usedSize = index == currentSlabIndex
                                               ? currentPointer - laterSlabBegin
                                               : computeSlabSize(index);
        __asan_poison_memory_region(laterSlabBegin, usedSize);
      }
    }

    currentPointer = checkpoint.pointer;
    currentSlabIndex = checkpoint.slabIndex;
    endSentinel = reinterpret_cast<uintptr_t>(slabEnd) + 1;
  }

  --checkpointDepth;
}

void BumpAllocator::clear() {
  deallocateSlabs(/*beginIndex=*/0);
  deallocateCustomSizedSlabs();
  slabs.clear();
  customSizedSlabs.clear();
  activeCustomSlabs.clear();
  availableCustomSlabs.clear();
  currentPointer = nullptr;
  currentSlabIndex = noSlab;
  endSentinel = 0;
}
