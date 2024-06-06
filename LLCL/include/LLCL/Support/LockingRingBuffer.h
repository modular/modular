//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_LOCKINGRINGBUFFER_H
#define LLCL_SUPPORT_LOCKINGRINGBUFFER_H

#include "Support/Threading/Atomics.h"
#include "Support/Threading/SpinWaiter.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <llvm/ADT/SmallVector.h>
#include <memory>

namespace M::LLCL {

/// This class provides a locking ring buffer for concurrent access.
template <typename ItemType>
class LockingRingBuffer {

public:
  LockingRingBuffer(size_t size)
      : size(llvm::NextPowerOf2(size)),
        buffer(std::make_unique<ItemType[]>(this->size)) {
    assert(llvm::isPowerOf2_64(this->size) &&
           "Ring buffer size is not power of 2.");
  }

  ~LockingRingBuffer() {
    assert(readIndex == writeIndex &&
           "Cannot destroy a non-empty ring buffer!");
  }

  /// Enqueue adds the object to the circular buffer and returns true, or
  /// returns false if the buffer is full.
  ///
  /// On success, it takes ownership of the object, std::move'ing from it.
  bool enqueue(ItemType &item) {
    std::lock_guard<std::mutex> lock(mu);
    return enqueueInternal(item);
  }

  bool enqueue(llvm::ArrayRef<ItemType> items) {
    std::lock_guard<std::mutex> lock(mu);
    for (auto &item : items)
      if (!enqueueInternal(item))
        return false;
    return true;
  }

  /// Dequeue returns the stored item to the caller and release the ownership of
  /// the item. Returns a value-initialized `ItemType` if the buffer is empty.
  ItemType dequeue() {
    std::lock_guard<std::mutex> lock(mu);
    return dequeueInternal();
  }

  llvm::SmallVector<ItemType> dequeue(size_t count) {
    llvm::SmallVector<ItemType> items;
    items.reserve(count);

    std::lock_guard<std::mutex> lock(mu);
    for (size_t i = 0; i < count; ++i) {
      auto item = dequeueInternal();
      if (item)
        items.emplace_back(std::move(item));
      else
        break;
    }
    return items;
  }

private:
  LockingRingBuffer(const LockingRingBuffer &other) = delete;
  LockingRingBuffer &operator=(const LockingRingBuffer &other) = delete;

  size_t used() const { return writeIndex - readIndex; }

  bool enqueueInternal(ItemType &item) {
    // Make sure that the buffer is not full.
    if (used() >= size)
      return false;

    const auto idx = writeIndex++;
    // Effectively `buffer[idx % size]` when size is power of 2.
    buffer[idx & (size - 1)] = std::move(item);
    return true;
  }

  ItemType dequeueInternal() {
    // Make sure that the buffer is not empty.
    if (readIndex == writeIndex)
      return ItemType();

    const auto idx = readIndex++;
    // Effectively `buffer[curReadIndex % size]` when size is power of 2.
    auto ret = std::move(buffer[idx & (size - 1)]);
    return ret;
  }

  size_t size;
  std::unique_ptr<ItemType[]> buffer;
  std::mutex mu;

  uint64_t readIndex = 0;
  uint64_t writeIndex = 0;
};

} // namespace M::LLCL

#endif // LLCL_SUPPORT_LOCKINGRINGBUFFER_H
