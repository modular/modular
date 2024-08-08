//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "AsyncRT/Runtime/Runtime.h"

using namespace M::AsyncRT;

std::unique_ptr<Runtime>
RuntimeOptions::createRuntime(StringRef profileName) const {
  RuntimeOptions runtimeOptions; //{*this};
  switch (allocatorType) {
  case RuntimeOptions::AllocatorType::kMalloc:
    runtimeOptions.tcmallocAllocator = false;
    break;
  case RuntimeOptions::AllocatorType::kTCMalloc:
    runtimeOptions.tcmallocAllocator = true;
    break;
  case RuntimeOptions::AllocatorType::kLeakChecker:
    runtimeOptions.leakCheckedAllocator = true;
    break;
  case RuntimeOptions::AllocatorType::kProfiler:
    runtimeOptions.profilingAllocator = true;
    break;
  case RuntimeOptions::AllocatorType::kUseAfterFree:
#if defined(HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR)
    runtimeOptions.useAfterFreeAllocator = true;
#else
    llvm::errs() << "The use-after-free allocator is not available for this "
                    "target. Using the leak-checker runtime instead.";
    options.leakCheckedAllocator = true;
#endif
    break;
  }
  // runtimeOptions.workQueueType = getWorkQueueType();
  switch (getWorkQueueType()) {
  case RuntimeOptions::WorkQueueType::kDefault:
    assert(0 && "should be resolved");
    LLVM_FALLTHROUGH;
  case RuntimeOptions::WorkQueueType::kSingleThread:
    runtimeOptions.singleThreaded = true;
    break;
  case RuntimeOptions::WorkQueueType::kThreadPool:
    runtimeOptions.numThreads = numThreads;
    runtimeOptions.maxThreads = maxThreads;
    runtimeOptions.withAffinity = withAffinity;
    runtimeOptions.threadBusyWaitTime = threadBusyWaitTime;
#if MODULAR_PARANOID
    runtimeOptions.paranoid = paranoid;
#endif
    break;
  }
  runtimeOptions.profileFilename = profileName;
  runtimeOptions.profilerDebuginfo = profilerDebuginfo;
  return AsyncRT::createUniqueRuntime(runtimeOptions);
}
