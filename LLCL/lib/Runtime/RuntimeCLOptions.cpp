//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "LLCL/Runtime/Runtime.h"

using namespace M::LLCL;

std::unique_ptr<Runtime>
RuntimeWorkQueueCLOptions::createRuntime(StringRef profileName) const {
  RuntimeOptions runtimeOptions;
  switch (allocatorType) {
  case AllocatorType::kMalloc:
    break;
  case AllocatorType::kLeakChecker:
    runtimeOptions.leakCheckedAllocator = true;
    break;
  case AllocatorType::kProfiler:
    runtimeOptions.profilingAllocator = true;
    break;
  case AllocatorType::kUseAfterFree:
#if defined(HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR)
    runtimeOptions.useAfterFreeAllocator = true;
#else
    llvm::errs() << "The use-after-free allocator is not available for this "
                    "target. Using the leak-checker runtime instead.";
    runtimeOptions.leakCheckedAllocator = true;
#endif
    break;
  }
  switch (getWorkQueueType()) {
  case WorkQueueType::kDefault:
    assert(0 && "should be resolved");
  case WorkQueueType::kSingleThread:
    runtimeOptions.singleThreaded = true;
    break;
  case WorkQueueType::kThreadPool:
    runtimeOptions.numThreads = numThreads;
    runtimeOptions.threadBusyWaitTime =
        std::chrono::microseconds(threadBusyWaitTime);
#if MODULAR_PARANOID
    runtimeOptions.paranoid = paranoid;
#endif
    break;
  }
  runtimeOptions.profileFilename = profileName;
  return LLCL::createUniqueRuntime(runtimeOptions);
}

std::unique_ptr<Runtime> RuntimeCLOptions::createRuntime() const {
  return RuntimeWorkQueueCLOptions::createRuntime(getProfileFilename());
}
