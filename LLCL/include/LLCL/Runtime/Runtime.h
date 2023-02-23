//===----------------------------------------------------------------------===//
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

#ifdef NO_LLCL_RUNTIME
#error Unintended dependency on the LLCL Runtime
#endif

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Chain.h"
#include "Support/TimeProfiler.h"
#include "llvm/ADT/StringRef.h"
#include <atomic>

namespace M {
class Error;
}

namespace M::LLCL {
class Allocator;
class WorkQueue;

/// This represents one instance of the LLCL runtime, which can have multiple
/// threads, a private heap for data, and a way of reporting errors.  This is
/// also the natural unit for cancellation.
///
class Runtime final {
public:
  /// Construct runtime with allocator and workQueue. If profileFilename is
  /// non-empty then time profiling will be activated and the profile JSON
  /// will be written to that file.
  Runtime(std::unique_ptr<Allocator> allocator,
          std::unique_ptr<WorkQueue> workQueue, StringRef profileFilename = {});
  ~Runtime();

  /// Return a CompactRuntimePtr that identifies this Runtime instance.
  CompactRuntimePtr getCompactPtr() const {
    return CompactRuntimePtr(runtimeIndex);
  }

  /// Return a reference to a pre-allocated Chain value that is already ready.
  /// This can be used by logic that needs to flag that a side effect has
  /// already happened, without doing an extraneous memory allocation.
  const AsyncValueRef<Chain> &getReadyChain() const { return readyChain; }

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

  /// Get direct access to the low level WorkQueue.  You should typically
  /// interface with the higher level algorithms in Algorithms.h.
  WorkQueue *getWorkQueue() { return workQueue.get(); }

  //===--------------------------------------------------------------------===//
  // Cancel the current execution
  //===--------------------------------------------------------------------===//

  /// Cancel the current MEF Execution. This transitions this Runtime to the
  /// canceled state, which causes all asynchronously executing threads to be
  /// canceled when they check the cancellation state (e.g. in MEFExecutor).
  void cancelExecution(EncodedDiagnostic message);

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

  /// The 'signature' for the type id registration system the runtime depends
  /// on. This is expected to be unique for the running process. This can be
  /// used to catch, at runtime, accidental multiple definitions for Modular
  /// runtime statics across dynamic libraries / executables.
  intptr_t signature;

  /// These are the allocator and workQueue's that were configured by the client
  /// for this Runtime.
  std::unique_ptr<Allocator> allocator;
  std::unique_ptr<WorkQueue> workQueue;

  /// Filename into which time profiling should be written, or the empty
  /// string if disabled.
  std::string profileFilename;

  /// An active profiler used for the runtime, or nullopt if profiling is
  /// disabled. This is only set when profileFilename is non-empty.
  std::optional<TimeTraceProfiler> profiler;

  /// This is the index # for the runtime object created.  This is held by the
  /// CompactRuntimePtr.
  uint8_t runtimeIndex;

  /// This is a preallocated Chain value that is marked as ready, for use by
  /// getReadyChain.
  AsyncValueRef<Chain> readyChain;

  /// If execution is cancelled, this holds the error value to forward into the
  /// results of computations.
  std::atomic<AsyncValue *> cancelValue{nullptr};

  friend void checkUniqueRuntime(const Runtime &runtime);
};

/// In debug builds, assert the given runtime's 'signature' agrees with what
/// the host's idea of signature for its dynamic library / executable.
/// This can be used to catch, at runtime, accidental multiple definitions for
/// Modular runtime statics across dynamic libraries / executables.
inline void checkUniqueRuntime(const Runtime &runtime) {
  assert(runtime.signature == TypeID::getSignature() &&
         "It appears your process has statically linked the Modular Runtime "
         "multiple times across dynamic library / executable boundaries. "
         "Please don't do that.");
}

} // namespace M::LLCL

#endif // LLCL_RUNTIME_H
