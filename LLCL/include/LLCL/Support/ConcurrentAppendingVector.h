//===- LLCL/Support/ConcurrentAppendingVector.h -----------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConcurrentAppendingVector class, a thread-safe
// vector-like container whose only mutation operation is emplace_back.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_CONCURRENT_APPENDING_VECTOR_H
#define LLCL_SUPPORT_CONCURRENT_APPENDING_VECTOR_H

#include <mutex>
#include <vector>

namespace LLCL {

/// This is a sequential container that allows concurrent lock-free reads and
/// locked append operations.  It is designed for the usage pattern where
/// objects are appended once but are read many times.  Once added, elements are
/// immutable.
///
/// The key difference between this data structure and std::vector is that when
/// we re-allocate the underlying buffer, we do not free the previous buffer.
/// This allows us to implement reads with a single atomic load.  This implies
/// that this should only be used with small elements like pointers, and that
/// the elements must be copyable.
///
/// Example usage:
///
///    ConcurrentAppendingVector<T> vec;
///    size_t index1 = vec.emplace_back(args);
///    size_t index2 = vec.emplace_back(args);
///    const auto &t1 = vec[index1];
///    const auto &t2 = vec[index2];
///
/// Both readers and writers are allowed to be concurrent.
///
/// TODO: This could be much more efficient by getting rid of the std::vector's.
/// We could instead maintain our array of `std::pair<T*, size_t>` records
/// that contain allocated space and capacity, replacing allAllocatedVectors.
/// We could then make append be completely lock-free in the common case where
/// a reallocation isn't required.
///
template <typename T>
class ConcurrentAppendingVector {
public:
  // Initialize the vector with the given initial_capapcity
  explicit ConcurrentAppendingVector(size_t initialCapacity) : state(0ull) {
    // We need to keep track of all of the vectors we allocate over time so we
    // can destroy them in our destructor.  This does not support inserting more
    // than 2^32 elements.
    allAllocatedVectors.reserve(32);
    allAllocatedVectors.emplace_back();
    auto &v = allAllocatedVectors.back();
    v.reserve(std::max(static_cast<size_t>(1), initialCapacity));
  }

  const T &operator[](size_t index) const {
    auto curState = getState(std::memory_order_acquire);
    assert(index < curState.size && "invalid index");
    return allAllocatedVectors[curState.lastAllocated][index];
  }

  // Return the number of elements currently valid in this vector.  The vector
  // only grows, so this is conservative w.r.t. the execution of the current
  // thread.
  size_t size() const { return getState(std::memory_order_relaxed).size; }

  // Insert a new element at the end. If the current buffer is full, we allocate
  // a new buffer with twice as much capacity and copy the items in the
  // previous buffer over.
  //
  // Returns the index of the newly inserted item.
  template <typename... Args>
  size_t emplace_back(Args &&...args) {
    std::lock_guard<std::mutex> lock(mutex);

    auto &last = allAllocatedVectors.back();

    if (last.size() < last.capacity()) {
      // There is still room in the current vector without reallocation. Just
      // add the new element there.
      last.emplace_back(std::forward<Args>(args)...);

      // Increment the size of the concurrent vector.
      auto curState = getState(std::memory_order_relaxed);
      curState.size += 1;
      state.store(curState.encode(), std::memory_order_release);

      return curState.size - 1; // return insertion index
    }

    // There is no more room in the current vector without reallocation.
    // Allocate a new vector with twice as much capacity, copy the elements
    // from the previous vector, and set elements_ to point to the data of the
    // new vector.
    allAllocatedVectors.emplace_back();
    auto &newLast = allAllocatedVectors.back();
    auto &prev = *(allAllocatedVectors.rbegin() + 1);
    newLast.reserve(prev.capacity() * 2);
    assert(prev.size() == prev.capacity());

    // Copy over the previous vector to the new vector.
    newLast.insert(newLast.begin(), prev.begin(), prev.end());
    newLast.emplace_back(std::forward<Args>(args)...);

    // Increment the size of the concurrent vector and index of the last
    // allocated vector.
    auto curState = getState(std::memory_order_relaxed);
    curState.lastAllocated += 1;
    curState.size += 1;
    state.store(curState.encode(), std::memory_order_release);
    return curState.size - 1; // return insertion index
  }

private:
  // Concurrent vector state layout:
  // - Low 32 bits encode the index of the last allocated vector.
  // - High 32 bits encode the size of the concurrent vector.
  struct State {
    uint32_t lastAllocated; // index of last allocated vector
    uint32_t size;          // size of the concurrent vector

    uint64_t encode() const {
      return (uint64_t(size) << 32) | uint64_t(lastAllocated);
    }
  };

  State getState(std::memory_order memOrder) const {
    uint64_t curState = state.load(memOrder);
    return {uint32_t(curState), uint32_t(curState >> 32)};
  }

  /// Mutations of this atomic are used to enforce happens-before relationship
  /// between emplace_back and operator[].
  std::atomic<uint64_t> state;

  /// This mutex protects allAllocatedVectors.
  std::mutex mutex;
  std::vector<std::vector<T>> allAllocatedVectors;
};

} // namespace LLCL

#endif // LLCL_SUPPORT_CONCURRENT_APPENDING_VECTOR_H
