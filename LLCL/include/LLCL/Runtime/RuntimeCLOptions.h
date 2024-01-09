//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a basic set of command line options for setting up and
// configuring an M::LLCL::Runtime for tools to use.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_RUNTIMECLOPTIONS_H
#define LLCL_RUNTIME_RUNTIMECLOPTIONS_H

#include "LLCL/Support/GenericUniquePtrSet.h"
#include "LLCL/Support/Profiling.h"
#include "Support/CommandLine.h"
#include "Support/RCRef.h"
#include "llvm/Support/Threading.h"
#include <chrono>
#include <thread>
#include <type_traits>

namespace M::LLCL {

class Runtime;

/// Contains a number of command-line options that are shared among binaries
/// that use the LLCL Runtime and want configurability of Allocator, WorkQueue,
/// etc.
class RuntimeWorkQueueCLOptions {
private:
  llvm::cl::OptionCategory RuntimeWorkQueueOptionsCategory{
      "Runtime work queue command line options"};

  //===--------------------------------------------------------------------===//
  // Core Runtime configuration.
  //===--------------------------------------------------------------------===//
protected:
  enum class AllocatorType {
    /// Allocator that just calls malloc/free.
    kMalloc,
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

  // Enable HostAllocator types to be specified on the command line.
  llvm::cl::opt<WorkQueueType> workQueueType{
      "workqueue", llvm::cl::desc("Specify workqueue type:"),
      llvm::cl::values(
          clEnumValN(WorkQueueType::kDefault, "default",
                     "Auto-select based on # threads"),
          clEnumValN(WorkQueueType::kSingleThread, "single-thread",
                     "Work queue that only ever uses one thread"),
          clEnumValN(WorkQueueType::kThreadPool, "thread-pool",
                     "Default threaded work queue based on std::thread")),
      llvm::cl::init(WorkQueueType::kDefault),
      llvm::cl::cat(RuntimeWorkQueueOptionsCategory)};

  // Enable HostAllocator types to be specified on the command line.
  llvm::cl::opt<AllocatorType> allocatorType{
      "allocator", llvm::cl::desc("Specify allocator type:"),
      llvm::cl::values(
          clEnumValN(AllocatorType::kMalloc, "malloc", "System malloc/free"),
          clEnumValN(AllocatorType::kLeakChecker, "leak-checker",
                     "Allocator with leak checking"),
          clEnumValN(AllocatorType::kProfiler, "profiler",
                     "Allocator with profiling and leak checking"),
          clEnumValN(AllocatorType::kUseAfterFree, "use-after-free",
                     "Allocator to detect use-after-free errors. Not available "
                     "on all targets.")),
      llvm::cl::init(
#ifdef MODULAR_DEBUG
          AllocatorType::kLeakChecker
#else
          AllocatorType::kMalloc
#endif
          ),
      llvm::cl::cat(RuntimeWorkQueueOptionsCategory)};

  // Specify the number of threads. If `thread==1`, then we automatically set
  // our work queue to `WorkQueueType::kSingleThread`. Otherwise, we assume the
  // work queue is using a thread pool. The default number of threads is the
  // result of M::getNumThreads().
  llvm::cl::opt<size_t> numThreads{
      "num-threads",
      llvm::cl::desc(
          "Specify the number of threads to run the work queue items. If zero "
          "(default), will be chosen by heuristics."),
      llvm::cl::init(0), llvm::cl::cat(RuntimeWorkQueueOptionsCategory)};

  // Specify the amount of time a worker thread should spin for before sleeping.
  // The optimal value here depends on the system latency for thread sleep and
  // wake-up, as well as other external factors like how many other threadpools
  // are sharing the system.
  llvm::cl::opt<size_t> threadBusyWaitTime{
      "thread-busy-wait-time-us",
      llvm::cl::desc(
          "Specify the number of microseconds for threads to spin before "
          "locking. Zero indicates that threads should never spin."),
      llvm::cl::init(200), llvm::cl::cat(RuntimeWorkQueueOptionsCategory)};

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

#if MODULAR_PARANOID
  /// If true, and in a MODULAR_PARANOID build, perform additional (and
  /// very expensive!) runtime actions to make race conditions and other
  /// undefined behaviour more likely to be observed by unit tests.
  llvm::cl::opt<bool> paranoid{
      "paranoid", llvm::cl::desc("Turn on paranoid mode"),
      llvm::cl::init(false), llvm::cl::cat(RuntimeWorkQueueOptionsCategory)};
#endif

  /// Constructor allows to specify default work queue (e.g. to force always
  /// using single thread)
  RuntimeWorkQueueCLOptions(
      WorkQueueType defaultWorkQueue = WorkQueueType::kThreadPool)
      : defaultWorkQueue(defaultWorkQueue) {}

  WorkQueueType defaultWorkQueue;

public:
  /// Return the number of threads specified at the command-line.
  size_t getNumThreads() const { return numThreads; }

  std::chrono::microseconds getThreadBusyWaitTime() const {
    return std::chrono::microseconds(threadBusyWaitTime);
  }

  /// Explicitly tell runtime to use single threaded workqueue. This is useful
  /// in situations where computation is performed by some other runtime (for
  /// eg: ExternalFrameworks in benchmarking)
  void useSingleThreadedWorkqueue() {
    numThreads = 1;
    workQueueType = WorkQueueType::kSingleThread;
  }

  /// Create a Runtime based on the CL argument specifications.
  std::unique_ptr<Runtime> createRuntime(StringRef profileName = {}) const;
};

/// Contains a number of command-line options that are shared among binaries
/// that use the LLCL Runtime and want configurability of Allocator, WorkQueue,
/// stopping behavior, etc.
///
class RuntimeCLOptions : public RuntimeWorkQueueCLOptions {
  //===--------------------------------------------------------------------===//
  // Core Runtime configuration.
  //===--------------------------------------------------------------------===//
private:
  llvm::cl::OptionCategory RuntimeOptionsCategory{
      "Runtime command line options"};

  // Filename to hold the time profiling output (as JSON text).
  llvm::cl::opt<std::string> profileFilename{
      "time-profile",
      llvm::cl::desc(
          kIsProfilingEnabled
              ? "Specify the filename base for profiling output. The tracing "
                "data will be written to a file called \"<base>.time-trace\". "
                "This will be a JSON text in the standard profiling format. "
                "The events will be written to a text file called "
                "\"<base>.time-events.csv\". An empty filename base disables "
                "profiling (the default)."
              : "Specify the filename base for profiling output. WARNING: This "
                "option is ignored in this build. Rebuild with "
                "MODULAR_LLCL_MAX_PROFILING_LEVEL greater than 0 to enable "
                "it."),
      llvm::cl::init(""), llvm::cl::cat(RuntimeOptionsCategory)};

  // Returns the filename to hold the time profiling output (as JSON text).
  // Returns empty string if profiling is disabled.
  StringRef getProfileFilename() const {
    if constexpr (!kIsProfilingEnabled) {
      if (!profileFilename.empty())
        llvm::errs()
            << "WARNING: The --time-profile option was given but this build"
               " does not support profiling. Rebuild with "
               "MODULAR_LLCL_MAX_PROFILING_LEVEL greater than 0 to enable "
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

  // Returns true if profiling is enabled by command line flag.
  bool getProfilingEnabled() const {
    if constexpr (!kIsProfilingEnabled)
      return false;
    return !profileFilename.empty();
  }

public:
  /// Print information about the runtime configuration to standard out.
  void printRuntimeConfig() const {
    printf("runtime using ");
    switch (allocatorType) {
    case AllocatorType::kMalloc:
      printf("malloc");
      break;
    case AllocatorType::kLeakChecker:
      printf("leak check");
      break;
    case AllocatorType::kProfiler:
      printf("profiling");
      break;
    case AllocatorType::kUseAfterFree:
      printf("use-after-free");
      break;
    }
    printf(" allocator, and ");
    switch (getWorkQueueType()) {
    case WorkQueueType::kDefault:
      assert(0 && "should be resolved");
    case WorkQueueType::kSingleThread:
      printf("single thread work queue");
      break;
    case WorkQueueType::kThreadPool:
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

  /// Create a Runtime based on the CL argument specifications.
  std::unique_ptr<Runtime> createRuntime() const;

  //===--------------------------------------------------------------------===//
  // Behavior indicating what to do when a test fails.
  //===--------------------------------------------------------------------===//

private:
  enum class OnFailure {
    kContinue,
    kExit,
  };

public:
  /// Set the behavior of executors if one of the functions they should run
  /// returns with an error. E.g. Set to `continue` for diagnostic verification.
  llvm::cl::opt<OnFailure> onFailure{
      "on-failure",
      llvm::cl::desc("Behavior in case an executed function returns with an "
                     "error. Ignored if there is only one function executed."),
      llvm::cl::values(
          clEnumValN(OnFailure::kContinue, "continue", "System malloc/free"),
          clEnumValN(OnFailure::kExit, "exit", "Allocator with leak checking")),
      llvm::cl::init(OnFailure::kExit)};

  /// Returns whether an executor should stop when a model returns an error.
  bool stopOnFirstError() const { return onFailure == OnFailure::kExit; }
};

} // namespace M::LLCL

#endif // LLCL_RUNTIME_RUNTIMECLOPTIONS_H
