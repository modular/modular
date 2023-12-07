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
  case AllocatorType::kUseAfterFree:
#ifdef HAVE_MODULAR_USE_AFTER_FREE_ALLOCATOR
    allocator = createUseAfterFreeAllocator();
#else
    llvm::errs() << "The use-after-free allocator is not available for this "
                    "target. Using the leak-checker runtime instead.";
    allocator = createLeakCheckAllocator(createMallocAllocator());
#endif
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
    // Let the ThreadPoolWorkQueue decide on an appropriate number of threads
    // if it is zero. It may be more sophisticated than getNumThreads().
    workQueue = createThreadPoolWorkQueue(
        numThreads, std::chrono::microseconds(threadBusyWaitTime)
#if MODULAR_PARANOID
                        ,
        paranoid
#endif
    );
    break;
  }
  return std::make_unique<Runtime>(std::move(allocator), std::move(workQueue),
                                   profileName);
}

std::unique_ptr<Runtime> RuntimeCLOptions::createRuntime() const {
  return RuntimeWorkQueueCLOptions::createRuntime(getProfileFilename());
}
