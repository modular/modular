//===- LLCL/Runtime/RuntimeCLOptions.h ------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a basic set of command line options for setting up and
// configuring an LLCL::Runtime for tools to use.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_RUNTIMECLOPTIONS_H
#define LLCL_RUNTIME_RUNTIMECLOPTIONS_H

#include "LLCL/Runtime/Runtime.h"
#include "Support/CommandLine.h"
#include <thread>
#include <type_traits>

namespace LLCL {

/// Contains a number of command-line options that are shared among binaries
/// that use the LLCL Runtime and want configurability of Allocator, WorkQueue,
/// stopping behavior, etc.
///
class RuntimeCLOptions {
  //===--------------------------------------------------------------------===//
  // Core Runtime configuration.
  //===--------------------------------------------------------------------===//
private:
  enum class AllocatorType {
    /// Allocator that just calls malloc/free.
    kMalloc,
    /// Allocator that does leak checking.
    kLeakChecker,
    /// Allocator that does profiling (and leak checking).
    kProfiler,
  };

  enum class WorkQueueType {
    /// Autosense work queue type based on # threads.
    kDefault,
    /// Workqueue that only ever uses one thread.
    kSingleThread,
    /// Default thread pool that uses std::thread and semaphores.
    kThreadPool,
    /// "No Interesting Name" Experimental Work Queue
    kNINE,
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
                     "Default threaded work queue based on std::thread"),
          clEnumValN(WorkQueueType::kNINE, "nine",
                     "'No Interesting Name' Experimental Work Queue")),
      llvm::cl::init(WorkQueueType::kDefault)};

  // Enable HostAllocator types to be specified on the command line.
  llvm::cl::opt<AllocatorType> allocatorType{
      "allocator", llvm::cl::desc("Specify allocator type:"),
      llvm::cl::values(
          clEnumValN(AllocatorType::kMalloc, "malloc", "System malloc/free"),
          clEnumValN(AllocatorType::kLeakChecker, "leak-checker",
                     "Allocator with leak checking"),
          clEnumValN(AllocatorType::kProfiler, "profiler",
                     "Allocator with profiling and leak checking")),
      llvm::cl::init(AllocatorType::kLeakChecker)};

  // Specify the number of threads. If `thread==1`, then we automatically set
  // our work queue to `WorkQueueType::kSingleThread`. Otherwise, we assume the
  // work queue is using a thread pool. The default number of threads is the
  // result of std::thread::hardware_concurrency().
  llvm::cl::opt<size_t> numThreads{
      "num-threads",
      llvm::cl::desc(
          "Specify the number of threads to run the work queue items."),
      llvm::cl::init(0)};

  // Specify the busy-wait duration of thread-pool work queue.
  llvm::cl::opt<unsigned> busyWaitNs{
      "busy-wait-ns",
      llvm::cl::desc(
          "Specify thread-pool work queue busy-wait duration in nanoseconds"),
      llvm::cl::Hidden, llvm::cl::init(0)};

  // Return the workqueue type to use, resolving kDefault into a concrete kind.
  WorkQueueType getWorkQueueType() const {
    // The default behavior picks a thread count based on the -num-threads
    // command line setting, but can be overridden.
    if (workQueueType == WorkQueueType::kDefault)
      return numThreads == 1 ? WorkQueueType::kSingleThread
                             : WorkQueueType::kThreadPool;
    return workQueueType;
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
    case WorkQueueType::kNINE:
      printf("'no interesting name' experimental work queue");
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

  /// Return the number of threads to use. This is always canonicalized to be
  /// non-zero.
  size_t getNumThreads() const {
    // If numThreads is 0 then return number of threads on the system.
    return numThreads == 0 ? std::thread::hardware_concurrency() : numThreads;
  }

  /// Create a Runtime based on the CL argument specifications.
  Runtime createRuntime() const {
    // Create the allocator based on command line settings.
    std::unique_ptr<Allocator> allocator;
    switch (allocatorType) {
    case AllocatorType::kMalloc:
      allocator = createMallocAllocator();
      break;
    case AllocatorType::kLeakChecker:
      allocator = createLeakCheckAllocator(createMallocAllocator());
      break;
    case AllocatorType::kProfiler:
      allocator = createProfilingAllocator(createMallocAllocator());
      break;
    }
    // Create the WorkQueue based on command line settings.
    std::unique_ptr<WorkQueue> workQueue;
    switch (getWorkQueueType()) {
    case WorkQueueType::kDefault:
      assert(0 && "should be resolved");
    case WorkQueueType::kSingleThread:
      workQueue = createSingleThreadWorkQueue();
      break;
    case WorkQueueType::kThreadPool:
      workQueue = createThreadPoolWorkQueue(getNumThreads(), busyWaitNs);
      break;
    case WorkQueueType::kNINE:
      workQueue = createThreadPoolWorkQueue(getNumThreads(), busyWaitNs);
      break;
    }
    return Runtime(std::move(allocator), std::move(workQueue));
  }

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

  //===--------------------------------------------------------------------===//
  // Helper methods.
  //===--------------------------------------------------------------------===//

  /// Run a lambda or other callable with a new Runtime instance configured
  /// according to the command line argument specification.  Encircle this with
  /// a AsyncValue leak checker to catch simple bugs in the test suite.
  template <typename BodyFn>
  auto runWithLeakCheckedRuntime(const char *testName, BodyFn bodyFn) const {
    // If we are leak checking, remember how many AsyncValue's we started with.
    ssize_t numStartingLiveAsyncValues = 0;
    if (AsyncValue::isAllocationTrackingEnabled())
      numStartingLiveAsyncValues = AsyncValue::getNumAllocatedInstances();

    // Check leak status on exit from scope.
    struct LeakChecker {
      const char *testName;
      ssize_t numStartingLiveAsyncValues;

      ~LeakChecker() { // Make sure we're not leaking AsyncValues.
        if (AsyncValue::isAllocationTrackingEnabled()) {
          ssize_t numLiveAsyncValues = AsyncValue::getNumAllocatedInstances();
          if (numLiveAsyncValues != numStartingLiveAsyncValues) {
            fprintf(
                stderr,
                "Evaluation of testcase '%s' leaked %d async values (before: "
                "%d, after: %d)!\n",
                testName, int(numLiveAsyncValues - numStartingLiveAsyncValues),
                int(numStartingLiveAsyncValues), int(numLiveAsyncValues));
            abort();
          }
        }
      }
    } checker{testName, numStartingLiveAsyncValues};

    // Execute the body with a new runtime, which is destroyed when the body is
    // done.
    return bodyFn(createRuntime());
  }
};

} // namespace LLCL

#endif // LLCL_RUNTIME_RUNTIMECLOPTIONS_H
