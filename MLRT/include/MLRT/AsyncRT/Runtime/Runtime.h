//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares top level "god object" that organizes AsyncRT thread pool,
// memory allocator, etc.
//
// This header file is intended to be a low-dependency header that other things
// compose on top of.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_RUNTIME_H
#define MLRT_ASYNCRT_RUNTIME_H

#include "MLRT/AsyncRT/Runtime/Allocator.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/AsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/CompactRuntimePtr.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/AsyncRT/Support/Chain.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/STLExtras.h"
#include "Support/StringExtras.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"

#include <atomic>
#include <memory>
#include <string>

namespace M {
class Error;
} // namespace M

namespace M::MLRT {
class Allocator;
class WorkQueue;

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

struct AllocatorOptions {
  bool leakCheckedAllocator = false;
  bool tcmallocAllocator = true;
  bool profilingAllocator = false;
  bool useAfterFreeAllocator = false;
};

/// Collects all the options which influence a runtime.
struct RuntimeOptions {
  enum class AllocatorType {
    /// Allocator that just calls malloc/free.
    kMalloc,
    /// Allocator that calls into tcmalloc.
    kTCMalloc,
    /// Allocator that does leak checking.
    kLeakChecker,
    /// Allocator that does profiling (and leak checking).
    kProfiler,
    /// Allocator that read/write protects every freed block
    /// to detect use-after-free errors without ASAN. Nat available
    /// on all targets.
    kUseAfterFree,
  };

  enum class WorkQueueType {
    /// Autosense work queue type based on # threads.
    kDefault,
    /// Workqueue that only ever uses one thread.
    kSingleThread,
    /// Default thread pool that uses std::thread and semaphores.
    kThreadPool,
  };

  enum class ProfilerDebuginfo {
    /// No debug info generated.
    kNoProfiler,
    /// Generating debug info for Linux `perf`.
    kPerfProfiler,
    /// Generate debug info by loading kernels as a shared library. Should work
    /// will all profilers.
    kSOProfiler
  };

  size_t numThreads = 0;
  size_t maxThreads = 0;
  bool singleThreaded = false;

  /// Filepath to write profile to, which enables profiling only if set.
  std::string profileFilename;

  /// Runtime configurable filter for profiling types (`Trace::Type`).
  /// Currently this only takes "type" into account and ignores "level".
  /// So any non-zero value enables the level, in other words `11111` and
  /// `22222` and `12121` all have the same effect. Set this in Runtime's ctor
  /// via RuntimeOptions.runtimeProfilingTypeMask.
  ///
  /// For example:
  ///
  /// MLRT::RuntimeOptions rtOpt;
  /// rtOpt.runtimeProfilingTypeMask = 1 << Trace::typeBitshift(Trace::kOther);
  /// auto rt = MLRT::getOrCreateRuntime(MLRT::RuntimeSource::Test,
  /// rtOpt);
  ///
  /// Creates a Runtime that will only record `kOther` type events.
  uint64_t runtimeProfilingTypeMask = Trace::kFullyEnabled;

  bool mainWillDonate = true;
  // TODO arekay - revert to time units
  //  std::chrono::microseconds threadBusyWaitTime = 200us;
  size_t threadBusyWaitTime = 200;
  // Affinity is disabled by default due to performance issues with multiple
  // processes. Can be enabled by MODULAR_ENABLE_AFFINITY environment variable,
  // which in turn can be overridden by --cpu-affinity CLI flag.
  bool withAffinity = []() {
    auto env = llvm::sys::Process::GetEnv("MODULAR_ENABLE_AFFINITY");
    return env.has_value() && M::isTrueLike(*env);
  }();
  std::string poolName = "🔥 Thread";
  bool leakCheckedAllocator = false;
  bool tcmallocAllocator = true;
  bool profilingAllocator = false;
  bool useAfterFreeAllocator = false;
  WorkQueueType workQueueType{RuntimeOptions::WorkQueueType::kDefault};

  AllocatorType allocatorType{
#ifdef MODULAR_DEBUG
      RuntimeOptions::AllocatorType::kLeakChecker
#else
      RuntimeOptions::AllocatorType::kMalloc
#endif
  };

  ProfilerDebuginfo profilerDebuginfo = ProfilerDebuginfo::kNoProfiler;
  WorkQueueType defaultWorkQueue;
  explicit RuntimeOptions(MLRT::RuntimeOptions::WorkQueueType wq =
                              MLRT::RuntimeOptions::WorkQueueType::kThreadPool)
      : defaultWorkQueue(wq) {}
  /// Explicitly tell runtime to use single threaded workqueue. This is useful
  /// in situations where computation is performed by some other runtime
  void useSingleThreadedWorkqueue() {
    numThreads = 1;
    workQueueType = RuntimeOptions::WorkQueueType::kSingleThread;
  }

  // Return the workqueue type to use, resolving kDefault into a concrete kind.
  WorkQueueType getWorkQueueType() const {
    // The default behavior picks a thread count based on the -num-threads
    // command line setting, but can be overridden. -num-threads=0 means using
    // the default work queue.
    if (workQueueType == WorkQueueType::kDefault) {
      if (numThreads == 0)
        return defaultWorkQueue;
      if (numThreads == 1)
        return WorkQueueType::kSingleThread;
      return WorkQueueType::kThreadPool;
    }
    return workQueueType;
  }

  StringRef getProfileFilename() const {
    if constexpr (!kIsProfilingEnabled) {
      if (!profileFilename.empty())
        llvm::errs()
            << "WARNING: The --time-profile option was given but this build"
               " does not support profiling. Rebuild with "
               "MODULAR_ASYNCRT_MAX_PROFILING_LEVEL greater than 0 to enable "
               "it.\n";
      return "";
    }
#ifdef MODULAR_DEBUG
    if (!profileFilename.empty())
      llvm::errs()
          << "WARNING: Using the --time-profile option in debug mode is"
             " not recommended due to increased overhead. Please use"
             " a release build.\n";
#endif
    return profileFilename;
  }

  // Temporary shim, remove once we separate the Allocator from the Runtime
  // Extract the Allocator-specific options from the RuntimeOptions into a
  // new struct.
  AllocatorOptions getAllocatorOptions() const {
    return {leakCheckedAllocator, tcmallocAllocator, profilingAllocator,
            useAfterFreeAllocator};
  }

  /// Print information about the runtime configuration to standard out.
  void printRuntimeConfig() const {
    printf("runtime using ");
    switch (allocatorType) {
    case RuntimeOptions::AllocatorType::kMalloc:
      printf("malloc");
      break;
    case RuntimeOptions::AllocatorType::kTCMalloc:
      printf("tcmalloc");
      break;
    case RuntimeOptions::AllocatorType::kLeakChecker:
      printf("leak check");
      break;
    case RuntimeOptions::AllocatorType::kProfiler:
      printf("profiling");
      break;
    case RuntimeOptions::AllocatorType::kUseAfterFree:
      printf("use-after-free");
      break;
    }
    printf(" allocator, and ");
    switch (getWorkQueueType()) {
    case RuntimeOptions::WorkQueueType::kDefault:
      assert(0 && "should be resolved");
      printf("potential assertion failure");
      break;
    case RuntimeOptions::WorkQueueType::kSingleThread:
      printf("single thread work queue");
      break;
    case RuntimeOptions::WorkQueueType::kThreadPool:
      printf("thread pool work queue");
      break;
    }

    switch (numThreads) {
    case 0:
      printf(" with autosensed threads.\n");
      break;
    default:
      printf(" with %d thread%s.\n", (int)numThreads, &"s"[numThreads == 1]);
      break;
    }
  }

  /// Return the number of threads specified at the command-line.
  size_t getNumThreads() const { return numThreads; }

  /// Equality for all fields that affect runtime behavior.
  bool operator==(const RuntimeOptions &other) const;
  bool operator!=(const RuntimeOptions &other) const {
    return !(*this == other);
  }

  /// Create a copy of the RuntimeOptions.
  RuntimeOptions copy() const;

  RuntimeOptions &forDebug() {
    singleThreaded = true;
    leakCheckedAllocator = true;
    return *this;
  }

  RuntimeOptions &withMainWillNotDonate(bool mainWillNotDonate = true) {
    this->mainWillDonate = !mainWillNotDonate;
    return *this;
  }

  RuntimeOptions &withCPUAffinity(bool cpuAffinity = true) {
    this->withAffinity = cpuAffinity;
    return *this;
  }

  RuntimeOptions &
  withLeakCheckedAllocator(bool newLeakCheckedAllocator = true) {
    leakCheckedAllocator = newLeakCheckedAllocator;
    return *this;
  }

  RuntimeOptions &withTCMallocAllocator(bool newTcmallocAllocator = true) {
    tcmallocAllocator = newTcmallocAllocator;
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

  RuntimeOptions &withMaxThreads(size_t newMaxThreads) {
    maxThreads = newMaxThreads;
    return *this;
  }

  RuntimeOptions &withProfileFilename(StringRef newProfileFilename) {
    profileFilename = newProfileFilename;
    return *this;
  }
};

/// Indicates how the Runtime was created, for diagnostics and tracing.
enum class RuntimeSource {
  /// Created by M::Context.
  MaxContext,
  /// Created by Mojo stdlib / CompilerRT.
  MojoStdlib,
  /// Created for CPU device context.
  CPUDeviceContext,
  /// Created for unit tests, benchmarks, or test harnesses.
  Test,
};

/// This represents one instance of the AsyncRT runtime, which can have multiple
/// threads, a private heap for data, a way of reporting errors, and other
/// context objects. This is also the natural unit for task cancellation.
///
/// Runtime is reference-counted so that RuntimeRef can be RCRef<Runtime> and
/// support shared ownership. It inherits ReferenceCounted and must only be
/// destroyed via dropRef().
class Runtime final : public M::ReferenceCounted<Runtime> {
public:
  /// Construct runtime with the already reserved runtimePtr, and already
  /// created allocator and workQueue. The work queue must have been constructed
  /// with the same runtimePtr.
  ///
  /// \p source indicates how the runtime was created (for diagnostics).
  /// If profileFilename is non-empty then time profiling will be activated
  /// and the profile JSON and text will be written to files with that prefix.
  Runtime(CompactRuntimePtr runtimePtr, std::unique_ptr<Allocator> allocator,
          std::unique_ptr<WorkQueue> workQueue, RuntimeSource source,
          StringRef profileFilename = {},
          uint64_t runtimeProfilingTypeMask = Trace::kFullyEnabled,
          RuntimeOptions::ProfilerDebuginfo profilerDebuginfo =
              RuntimeOptions::ProfilerDebuginfo::kNoProfiler);
  ~Runtime();

  /// How this runtime was created (for diagnostics).
  RuntimeSource getSource() const { return source; }

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
  /// items on behalf of the runtime). If no runtime has been associated with
  /// this thread but a global runtime exists, automatically associates this
  /// thread with it. Returns null only if no global runtime exists at all.
  static Runtime *getCurrentRuntimeOrNull();

  //===--------------------------------------------------------------------===//
  // Profiling
  //===--------------------------------------------------------------------===//

  /// Return a reference to the profiler instance, if its been initialized.
  std::optional<TimeTraceProfiler> &getProfiler() { return profiler; }

  /// Which profiler should we generate debug information for.
  RuntimeOptions::ProfilerDebuginfo getProfilerDebuginfo() const {
    return profilerDebuginfo;
  }

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level allocator.
  Allocator *getAllocator() { return allocator.get(); }

  /// Returns the current runtime allocator. This assumes that a global
  /// allocator is present and would assert otherwise.
  static Allocator *getCurrentAllocator() {
    auto rt = Runtime::getCurrentRuntimeOrNull();
    assert(rt &&
           "a global runtime must be set before getting the current allocator");
    return rt->getAllocator();
  }

  //===--------------------------------------------------------------------===//
  // Concurrency
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level WorkQueue.  You should typically
  /// interface with the higher level algorithms in Algorithms.h.
  WorkQueue *getWorkQueue() { return workQueue.get(); }

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

  /// An active profiler used for the runtime, or nullopt if profiling is
  /// disabled. This is only set when profileFilename is non-empty.
  std::optional<TimeTraceProfiler> profiler;

  /// Should the runtime output debug info for `perf`.
  RuntimeOptions::ProfilerDebuginfo profilerDebuginfo;

  /// This is the index # for the runtime object created.  This is held by the
  /// CompactRuntimePtr.
  uint8_t runtimeIndex;

  /// How this runtime was created (for diagnostics).
  RuntimeSource source;

  /// This is a preallocated Chain value that is marked as ready, for use by
  /// getReadyChain.
  AsyncValueRef<Chain> readyChain;

  friend void checkUniqueRuntime(const Runtime &runtime);
};

//===----------------------------------------------------------------------===//
// Runtime construction
//===----------------------------------------------------------------------===//

/// Creates a suitable allocator given the options.
std::unique_ptr<Allocator>
getAllocator(const AllocatorOptions &options = AllocatorOptions());

using RuntimeRef = M::RCRef<Runtime>;

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

} // namespace M::MLRT

#endif // MLRT_ASYNCRT_RUNTIME_H
