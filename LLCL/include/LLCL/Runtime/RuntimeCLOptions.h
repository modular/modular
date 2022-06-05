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

#include "LLCL/Runtime/AllocatorType.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommandLine.h"
#include <type_traits>

namespace LLCL {

/// Contains a number of command-line options that are shared among most of our
/// binaries
struct RuntimeCLOptions {

  // Specify the number of threads. If `thread==1`, then we automatically set
  // our work queue to `WorkQueueType::kSingleThread`. Otherwise, we assume the
  // work queue is using a thread pool. The default number of threads is the
  // result of std::thread::hardware_concurrency().
  llvm::cl::opt<size_t> numThreads{
      "num-threads",
      llvm::cl::desc("Specify the number of threads in the threadpool"),
      llvm::cl::init(0)};

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

  Runtime createRuntime() const {
    return Runtime(getAllocator(allocatorType), getWorkQueue(numThreads));
  }

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
