//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "LLCL/Runtime/Runtime.h"

using namespace M::LLCL;

size_t RuntimeOptions::canonicalizeNumThreads(size_t numThreads) {
  if (numThreads)
    return numThreads;
  return M::getNumPhysicalCores();
}

std::unique_ptr<Runtime>
RuntimeOptions::createRuntime(StringRef profileName) const {
  RuntimeOptions runtimeOptions; //{*this};
  switch (allocatorType) {
  case RuntimeOptions::AllocatorType::kMalloc:
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
  case RuntimeOptions::WorkQueueType::kSingleThread:
    runtimeOptions.singleThreaded = true;
    break;
  case RuntimeOptions::WorkQueueType::kThreadPool:
    runtimeOptions.numThreads = numThreads;
    runtimeOptions.withAffinity = withAffinity;
    runtimeOptions.threadBusyWaitTime = threadBusyWaitTime;
#if MODULAR_PARANOID
    runtimeOptions.paranoid = paranoid;
#endif
    break;
  }
  runtimeOptions.profileFilename = profileName;
  return LLCL::createUniqueRuntime(runtimeOptions);
}
