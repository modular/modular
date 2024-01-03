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
#include "LLCL/Support/GenericUniquePtrSet.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>

namespace M {
class Error;
}

namespace M::LLCL {
class Allocator;
class WorkQueue;

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

/// This represents one instance of the LLCL runtime, which can have multiple
/// threads, a private heap for data, a way of reporting errors, and other
/// context objects. This is also the natural unit for task cancellation.
class Runtime final {
public:
  /// Construct runtime with the already reserved runtimePtr, and already
  /// created allocator and workQueue. The work queue must have been constructed
  /// with the same runtimePtr.
  ///
  /// If profileFilename is non-empty then time profiling will be activated
  /// and the profile JSON and text will be written to files with that prefix.
  Runtime(CompactRuntimePtr runtimePtr, std::unique_ptr<Allocator> allocator,
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

  /// Returns the runtime managing the work queue to which the callers thread
  /// is associated (ie the callers thread is either a worker thread for that
  /// runtime or is a 'main' thread which has donated itself to running work
  /// items on behalf of the runtime). Returns null if no such runtime has been
  /// associated.
  static Runtime *getCurrentRuntimeOrNull() {
    return CompactRuntimePtr::getCurrentRuntime().getOrNull();
  }

  /// As for getCurrentRuntimeOrNull, but assert fail if no runtime is
  /// associated.
  static Runtime &getCurrentRuntime() {
    Runtime *runtime = getCurrentRuntimeOrNull();
    assert(runtime && "no runtime is associated with the current thread");
    return *runtime;
  }

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
  // Context Objects
  //===--------------------------------------------------------------------===//

  /// Transfers ptr into the context object set.
  template <typename T>
  void setContext(std::unique_ptr<T> ptr) {
    contextObjects.set<T>(std::move(ptr));
  }

  /// Emplaces a new object of type T into the context object set and returns a
  /// reference to it.
  template <typename T, typename... Args>
  T &emplaceContext(Args &&...args) {
    return contextObjects.emplace<T, Args...>(std::forward<Args>(args)...);
  }

  /// Returns a reference to the object of type T held by the context object
  /// set. If it does not contain such an object, emplaces a new object and
  /// returns a reference to it.
  template <typename T, typename... Args>
  T &emplaceContextIfMissing(Args &&...args) {
    return contextObjects.emplaceIfMissing<T, Args...>(
        std::forward<Args>(args)...);
  }

  /// Returns a pointer to the object of type T held by the context object set.
  /// If it does not contain such an object, calls the creator function to
  /// create one and install. Returns any error the creator function returns.
  template <typename T>
  ErrorOr<T *> createContextIfMissing(
      llvm::unique_function<ErrorOr<std::unique_ptr<T>>()> creator) {
    return contextObjects.createIfMissing<T>(std::move(creator));
  }

  /// Returns a pointer to the context object of type T held by the context
  /// object set, or nullptr if no such object exists.
  template <typename T>
  T *getContext() {
    return contextObjects.get<T>();
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

  /// Set of 'context objects' owned by this runtime.
  GenericUniquePtrSet contextObjects;

  friend void checkUniqueRuntime(const Runtime &runtime);
  friend void checkKnownCallingThread(const Runtime &runtime);
};

/// Collects all the options which influence a runtime.
struct RuntimeOptions {
  size_t numThreads = 0;
  bool singleThreaded = false;
  StringRef profileFilename = {};
  bool mainWillDonate = true;
  std::chrono::microseconds threadBusyWaitTime = 200us;
  std::string_view poolName = "🔥 Thread";
  bool paranoid = false;
  bool leakCheckedAllocator = false;
  bool profilingAllocator = false;
  bool useAfterFreeAllocator = false;

  RuntimeOptions &forDebug() {
    singleThreaded = true;
    leakCheckedAllocator = true;
    return *this;
  }

  RuntimeOptions &withMainWillNotDonate(bool mainWillNotDonate = true) {
    this->mainWillDonate = !mainWillNotDonate;
    return *this;
  }

  RuntimeOptions &
  withLeakCheckedAllocator(bool newLeakCheckedAllocator = true) {
    leakCheckedAllocator = newLeakCheckedAllocator;
    return *this;
  }

  RuntimeOptions &withProfilingAllocator(bool newProfilingAllocator = true) {
    profilingAllocator = newProfilingAllocator;
    return *this;
  }

  RuntimeOptions &withSingleThreaded(bool newSingleThreaded = true) {
    singleThreaded = newSingleThreaded;
    return *this;
  }

  RuntimeOptions &withNumThreads(size_t newNumThreads) {
    numThreads = newNumThreads;
    return *this;
  }

  RuntimeOptions &withProfileFilename(StringRef newProfileFilename) {
    profileFilename = newProfileFilename;
    return *this;
  }
};

//===----------------------------------------------------------------------===//
// Runtime construction
//===----------------------------------------------------------------------===//

/// Creates a runtime with the given options, on the assumption the caller
/// is not within any outer runtime's thread (main or worker).
///
/// Consider using createRuntimeIfNeeded if it is possible an existing
/// runtime has already been established by an outer context, such as
/// within the Modular Execution Engine, and the caller is not particular
/// about the runtime options.
///
/// Consider using createNestedRuntime if it is possible an existing runtime
/// has already been established by an outer context, yet the caller must
/// use a runtime with the given options.
std::unique_ptr<Runtime>
createUniqueRuntime(const RuntimeOptions &options = RuntimeOptions());

/// Creates a runtime with the given options, where it is legal for the caller
/// to be within an outer runtime's thread (main or worker).
std::unique_ptr<Runtime>
createNestedRuntime(const RuntimeOptions &options = RuntimeOptions());

/// Returns the current runtime for the caller's thread (main or worker). If
/// no such runtime has been associated, creates a runtime with the given
/// options.
ConditionallyOwnedPointer<Runtime>
createRuntimeIfNeeded(const RuntimeOptions &options = RuntimeOptions());

//===----------------------------------------------------------------------===//
// Debugging helpers
//===----------------------------------------------------------------------===//

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
