//===- LLCL/Runtime/Runtime.h -----------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares top level "god object" that organizes LLCL thread pool,
// memory allocator, etc.
//
// This header file is intended to be a low-dependency header that other things
// compose on top of.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_H
#define LLCL_RUNTIME_H

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Runtime/WorkQueue.h"

namespace M {
class Error;
}

namespace LLCL {
class Allocator;
class WorkQueue;
using TaskFunction = llvm::unique_function<void()>;

/// This represents one instance of the LLCL runtime, which can have multiple
/// threads, a private heap for data, and a way of reporting errors.  This is
/// also the natural unit for cancelation.
///
class Runtime final {
public:
  // TODO: Diagnostics.
  Runtime(std::unique_ptr<Allocator> allocator,
          std::unique_ptr<WorkQueue> workQueue);
  ~Runtime();

  /// Return a CompactRuntimePtr that identifies this Runtime instance.
  CompactRuntimePtr getCompactPtr() const {
    return CompactRuntimePtr(runtimeIndex);
  }

  /// Return a reference to a pre-allocated Chain value that is already ready.
  /// This can be used by logic that needs to flag that a side effect has
  /// already happened, without doing an extraneous memory allocation.
  AsyncValueRef<Chain> getReadyChain() const;

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level allocator.
  Allocator *getAllocator() { return allocator.get(); }

  /// Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment) {
    return allocator->allocateBytes(size, alignment);
  }

  /// Deallocate the specified pointer that had the specified size.
  void deallocateBytes(void *ptr, size_t size) {
    return allocator->deallocateBytes(ptr, size);
  }

  /// Allocate memory for one or more entries of type T.
  template <typename T>
  T *allocate(size_t numElements = 1) {
    return static_cast<T *>(allocateBytes(sizeof(T) * numElements, alignof(T)));
  }

  /// Deallocate the memory for one or more entries of type T.
  template <typename T>
  void deallocate(T *ptr, size_t numElements) {
    deallocateBytes(ptr, sizeof(T) * numElements);
  }

  /// Allocate and initialize an object of type T.
  template <typename T, typename... Args>
  T *construct(Args &&...args) {
    T *buf = allocate<T>();
    return new (buf) T(std::forward<Args>(args)...);
  }

  /// Destruct and deallocate space for one or more object of type T.
  template <typename T>
  void destroyAndDeallocate(T *ptr, size_t numElements = 1) {
    for (size_t i = 0; i != numElements; ++i)
      ptr[i].~T();
    deallocate(ptr, numElements);
  }

  //===--------------------------------------------------------------------===//
  // Concurrency
  //===--------------------------------------------------------------------===//

  /// Enqueue a block of work. Thread-safe.
  void addTask(TaskFunction work) { workQueue->addTask(std::move(work)); }

  /// Block until the specified values are ready.  This should not be called by
  /// a thread managed by our work queue.
  void await(llvm::ArrayRef<RCRef<AsyncValue>> values);

  /// Block until the system is quiescent (no pending/inflight work).
  ///
  /// This should not be called by a thread managed by the work queue.
  void quiesce() { workQueue->quiesce(); }

  //===--------------------------------------------------------------------===//
  // Cancel the current execution
  //===--------------------------------------------------------------------===//

  /// Cancel the current BEF Execution. This transitions this Runtime to the
  /// canceled state, which causes all asynchronously executing threads to be
  /// canceled when they check the cancellation state (e.g. in BEFExecutor).
  void cancelExecution(M::Error message);

  /// restartFromCancellation() transitions Runtime from the canceled state to
  /// the normal execution state.
  void restartFromCancellation();

  /// When this Runtime is in a canceled state, getCancelValue() returns a
  /// non-null AsyncValue containing the message for the cancellation.
  /// Otherwise, it returns nullptr.
  AsyncValue *getCancelValue() const {
    return cancelValue.load(std::memory_order_acquire);
  }

private:
  Runtime(const Runtime &) = delete;
  void operator=(const Runtime &) = delete;

  /// These are the allocator and workQueue's that were configured by the client
  /// for this Runtime.
  const std::unique_ptr<Allocator> allocator;
  const std::unique_ptr<WorkQueue> workQueue;

  /// This is the index # for the runtime object created.  This is held by the
  /// CompactRuntimePtr.
  uint8_t runtimeIndex;

  /// This is a preallocated Chain value that is marked as ready, for use by
  /// getReadyChain.
  AsyncValue *const readyChain;

  /// If execution is cancelled, this holds the error value to forward into the
  /// results of computations.
  std::atomic<AsyncValue *> cancelValue{nullptr};
};

} // namespace LLCL

#endif // LLCL_RUNTIME_H
