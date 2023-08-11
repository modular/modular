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

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/GenericUniquePtr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>

namespace M {
class Error;
}

namespace M::LLCL {
class Allocator;
class WorkQueue;

/// This represents one instance of the LLCL runtime, which can have multiple
/// threads, a private heap for data, a way of reporting errors, and other
/// global context objects. This is also the natural unit for task cancellation.
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
  // Profiling
  //===--------------------------------------------------------------------===//

  /// Return a reference to the profiler instance, if its been initialized.
  std::optional<TimeTraceProfiler> &getProfiler() { return profiler; }

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level allocator.
  Allocator *getAllocator() { return allocator.get(); }

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

  //===--------------------------------------------------------------------===//
  // Contexts
  //===--------------------------------------------------------------------===//

  /// Emplaces a new global context object of type T into the runtime's set of
  /// contexts and returns a reference to it. The runtime can hold at
  /// most one context object per T. The returned reference is stable for the
  /// life of the runtime. Thread safe, though the caller is responsible for
  /// thread safe access to the context object itself.
  template <typename T, typename... Args>
  T &emplaceContext(Args &&...args) {
    std::lock_guard<std::mutex> lock(mu);
    auto genericPtr = makeGenericUniquePtr<T>(std::forward<Args>(args)...);
    auto denseIndex = genericPtr.getTypeID().getDenseIndex();
    assert(!contexts.contains(denseIndex) &&
           "Runtime already holds context of type");
    T &result = *genericPtr.template get<T>();
    contexts.insert({denseIndex, std::move(genericPtr)});
    return result;
  }

  /// Returns a reference to the global context object of type T held by the
  /// runtime if it exists, or otherwise it emplaces a new global context
  /// object and returns a reference to it. The returned reference is stable for
  /// the life of the runtime. Thread safe, though the caller is responsible for
  /// thread safe access to the context object itself.
  template <typename T, typename... Args>
  T &emplaceContextIfMissing(Args &&...args) {
    std::lock_guard<std::mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = contexts.find(denseIndex);
    if (itr == contexts.end()) {
      auto genericPtr = makeGenericUniquePtr<T>(std::forward<Args>(args)...);
      T &result = *genericPtr.template get<T>();
      contexts.insert({denseIndex, std::move(genericPtr)});
      return result;
    } else {
      return *(itr->second.template get<T>());
    }
  }

  /// Returns a pointer to the global context object of type T held by the
  /// runtime, or nullptr if no such object is held. Thread safe, though the
  /// caller is responsible for thread safe access to the context object itself.
  template <typename T>
  T *getContext() {
    std::lock_guard<std::mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = contexts.find(denseIndex);
    if (itr == contexts.end())
      return nullptr;
    return itr->second.template get<T>();
  }

  /// If the runtime does not already hold a global context object of type T,
  /// calls the creator function to create one and installs it. Returns either
  /// the existing or freshly created object. Returns any error the creator
  /// function returns. Thread safe, though the caller is responsible for
  /// thread safe access to the context object itself.
  template <typename T>
  ErrorOr<T *> createContextIfMissing(
      llvm::unique_function<ErrorOr<std::unique_ptr<T>>()> creator) {
    std::lock_guard<std::mutex> lock(mu);
    auto denseIndex = TypeID::get<T>().getDenseIndex();
    auto itr = contexts.find(denseIndex);
    if (itr != contexts.end())
      return itr->second.template get<T>();
    ErrorOr<std::unique_ptr<T>> errOr = creator();
    if (errOr.isError())
      return errOr.takeError();
    T *result = errOr->get();
    GenericUniquePtr genericPtr;
    genericPtr.reset(std::move(*errOr));
    contexts.insert({denseIndex, std::move(genericPtr)});
    return result;
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

  /// Protects contexts.
  std::mutex mu;

  /// A map from globally unique type identifiers TypeID::get<T>() (using
  /// their 'dense index' form) to GenericUniquePtr holding the global context
  /// object of type T.
  DenseMap<size_t, GenericUniquePtr> contexts;

  friend void checkUniqueRuntime(const Runtime &runtime);
  friend void checkKnownCallingThread(const Runtime &runtime);
};

/// In debug builds, assert the given runtime's 'signature' agrees with what
/// the host's idea of signature for its dynamic library / executable.
/// This can be used to catch, at runtime, accidental multiple definitions for
/// Modular runtime statics across dynamic libraries / executables.
inline void checkUniqueRuntime(const Runtime &runtime) {
  assert(runtime.signature ==
             (TypeID::getSignature() ^ CompactRuntimePtr::getSignature()) &&
         "It appears your process has statically linked the Modular Runtime "
         "multiple times across dynamic library / executable boundaries. "
         "Please don't do that.");
}

// In debug builds, assert the caller is one of threads managed by the given
// runtime's work queue. The thread may be an actual worker thread or the
// 'main' thread (if the work queue has that notion).
inline void checkKnownCallingThread(const Runtime &runtime) {
  assert(!runtime.workQueue->callerIsForeign() &&
         "Attempting to process work on a 'foreign' thread. Are you missing an "
         "addTask?");
}

} // namespace M::LLCL

#endif // LLCL_RUNTIME_H
