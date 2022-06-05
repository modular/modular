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
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace LLCL {

/// Contains a number of command-line options that are shared among most of our
/// binaries
struct RuntimeCLOptions {
  // Specify the number of threads. If `thread==1`, then we automatically set
  // our work queue to `WorkQueueType::kSingleThread`. Otherwise, we assume the
  // work queue is using a thread pool. The default number of threads is the
  // result of std::thread::hardware_concurrency().
  cl::opt<size_t> numThreads{
      "num-threads",
      cl::desc("Specify the number of threads in the threadpool"), cl::init(0)};

  // Enable HostAllocator types to be specified on the command line.
  cl::opt<AllocatorType> allocatorType{
      "allocator", cl::desc("Specify allocator type:"),
      cl::values(
          clEnumValN(AllocatorType::kMalloc, "malloc", "System malloc/free"),
          clEnumValN(AllocatorType::kLeakChecker, "leak-checker",
                     "Allocator with leak checking"),
          clEnumValN(AllocatorType::kProfiler, "profiler",
                     "Allocator with profiling and leak checking")),
      cl::init(AllocatorType::kLeakChecker)};

  Runtime createRuntime() const {
    return Runtime(getAllocator(allocatorType), getWorkQueue(numThreads));
  }
};

} // namespace LLCL

#endif // LLCL_RUNTIME_RUNTIMECLOPTIONS_H
